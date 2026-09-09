"""
Configuration, read once at startup.

Three sources, and the ordering is deliberate: a flag beats a file, a file beats
the environment, the environment beats a default. File-beats-environment is the
reverse of the usual convention and it will surprise people, so the CLI prints
where each value came from.

Import as:

import crux.infra.settings as isettn
"""

from __future__ import annotations

import os
import pathlib
import typing
from typing import Annotated, Any

import pydantic
import pydantic_settings

import crux.errors as cerrors
import crux.ports.reasoner as preason

# Fields that must never be settable from a checked-in file. The precedence order
# puts crux.toml above the environment, so without this a committed file would
# beat an operator's exported key.
_SECRET_FIELDS = frozenset({"api_key"})

# How the human-facing toml shape maps onto the flat field names. Flattening
# happens in a settings *source* rather than in a validator, because a validator
# runs on the already-merged mapping where a file value and a higher-priority
# environment value arrive together, and it would have to re-implement
# precedence to tell them apart.
_TOML_SECTIONS: dict[str, dict[str, str]] = {
    "tiers": {"reasoning": "model", "mechanical": "weak_model"},
    "fallback": {
        "models": "fallback_models",
        "retries": "fallback_retries",
        "deadline": "fallback_deadline",
    },
    "limits": {
        "max_tokens": "max_tokens",
        "request_timeout": "request_timeout",
        "max_retries": "max_retries",
        "max_passes": "max_passes",
        "max_rounds": "max_rounds",
        "max_questions_total": "max_questions_total",
        "max_questions_per_round": "max_questions_per_round",
    },
}


class ModelOptions(pydantic.BaseModel):
    """
    Per-model overrides for the call options that are otherwise global.

    Exists because a free tier and a paid one want different numbers. Groq's
    8,000 tokens per minute counts what you *reserve*, so a model there wants a
    small ``max_tokens``, while the same setting would truncate a frontier model
    mid-answer.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    request_timeout: float | None = None
    reasoning_effort: str | None = None


class ModelSpec(pydantic.BaseModel):
    """
    One model, with every call option already resolved.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    api_key: pydantic.SecretStr | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int = 8192
    request_timeout: float = 120.0
    reasoning_effort: str | None = None


class _CruxTomlSource(pydantic_settings.TomlConfigSettingsSource):
    """
    Reads ``crux.toml`` and flattens its sections onto the flat field names.

    The nesting is for humans. Making :class:`CruxSettings` itself nested would
    rename every environment variable to ``CRUX_TIERS__REASONING`` and break
    every existing ``.env``, so the shape is reconciled here instead and
    pydantic's own priority logic is left untouched.
    """

    def __call__(self) -> dict[str, Any]:
        """
        :return: Flat field names to values.
        :raises ConfigurationError: When the file tries to set a secret.
        """
        raw = super().__call__()
        flat: dict[str, Any] = {}

        for section, mapping in _TOML_SECTIONS.items():
            block = raw.get(section) or {}
            if not isinstance(block, dict):
                continue
            for key, field in mapping.items():
                if key in block:
                    flat[field] = block[key]

        operations = raw.get("operations") or {}
        if isinstance(operations, dict):
            for name, value in operations.items():
                if name in _OPERATION_FIELDS:
                    flat[_OPERATION_FIELDS[name]] = value

        models = raw.get("models")
        if isinstance(models, dict):
            flat["models"] = models

        # Anything at the top level that is a plain flat field name is honoured
        # too, so a one-line file does not need a section header.
        for key, value in raw.items():
            if key not in _TOML_SECTIONS and key not in ("operations", "models"):
                flat[key] = value

        leaked = _SECRET_FIELDS & set(flat)
        if leaked:
            raise cerrors.ConfigurationError(
                f"crux.toml must not set {sorted(leaked)}. This file takes "
                f"priority over the environment, so a checked-in secret here "
                f"would silently beat an operator's exported one. Keep "
                f"credentials in the environment."
            )
        return flat


_OPERATION_FIELDS: dict[str, str] = {
    operation: f"model_{operation}" for operation in typing.get_args(preason.Operation)
}


class CruxSettings(pydantic_settings.BaseSettings):
    """
    Everything crux reads from the environment, a file, or its caller.
    """

    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="CRUX_", env_file=".env", extra="ignore"
    )

    config_file: pathlib.Path | None = None
    """A ``crux.toml`` to read. The CLI finds one and passes it; a library host
    passes its own or none. crux never goes looking on its own behalf, for the
    same reason it does not mutate its host's environment."""

    # ## Tiers

    model: str = "claude-sonnet-5"
    weak_model: str | None = None
    """The mechanical tier. ``None`` means "same as the reasoning model", so a
    single ``CRUX_MODEL`` behaves exactly as it always has."""

    model_expand: str | None = None
    model_adjudicate: str | None = None
    model_phrase: str | None = None
    model_classify: str | None = None
    model_answer_counter: str | None = None
    model_draft: str | None = None
    model_select_pack: str | None = None

    # ## Fallback

    fallback_models: Annotated[tuple[str, ...], pydantic_settings.NoDecode] = ()
    """NoDecode because pydantic-settings JSON-decodes complex-typed fields
    inside the source, before any validator runs. Without it,
    ``CRUX_FALLBACK_MODELS=a,b,c`` raises a SettingsError rather than parsing,
    and a comma-separated list is the first thing anyone types."""
    fallback_retries: int = 1
    """Non-primary models get less patience. A fallback that is also rate limited
    is not the one to wait on."""

    fallback_deadline: float = 180.0
    """Wall clock for one completion across the whole chain, checked before each
    attempt starts. Without it, a chain of three at up to 65 seconds of backoff
    each is ten minutes for a single call."""

    # ## Per-model options, from the file only

    models: dict[str, ModelOptions] = pydantic.Field(default_factory=dict)

    # ## Global call options, the defaults a ModelOptions overlays

    api_key: pydantic.SecretStr | None = None
    api_base: str | None = None
    temperature: float | None = None
    max_tokens: int = 8192
    request_timeout: float = 120.0
    reasoning_effort: str | None = None
    max_retries: int = 3

    # ## Budgets

    max_passes: int = 3
    max_rounds: int = 3
    max_questions_total: int = 6
    max_questions_per_round: int = 3

    @pydantic.field_validator("fallback_models", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: Any) -> Any:
        """
        Accept a comma-separated string for the fallback chain.

        pydantic-settings decodes complex-typed fields from the environment as
        JSON, so ``CRUX_FALLBACK_MODELS=a,b,c`` raises rather than parsing. That
        is the first thing anyone would type.

        :param value: Whatever the source supplied.
        :return: A sequence pydantic can validate.
        """
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.PydanticBaseSettingsSource,
        env_settings: pydantic_settings.PydanticBaseSettingsSource,
        dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
        file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        """
        Order the sources so that a flag beats a file beats the environment.

        Earlier wins. The file sits above the environment because a repo that
        checks one in wants every contributor on the same models regardless of
        their shell; credentials are excluded from the file precisely so that
        this ordering stays safe.

        :param settings_cls: The settings class being built.
        :param init_settings: Keyword arguments, where the CLI puts its flags.
        :param env_settings: The process environment.
        :param dotenv_settings: The ``.env`` file.
        :param file_secret_settings: Docker-style secret files.
        :return: The sources, highest priority first.
        """
        path = _requested_config_file(init_settings)
        if path is None:
            return (init_settings, env_settings, dotenv_settings, file_secret_settings)
        return (
            init_settings,
            _CruxTomlSource(settings_cls, toml_file=path),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    def operation_overrides(self) -> dict[preason.Operation, str]:
        """
        Collect the per-operation escape hatches that are actually set.

        :return: Operation name to model name, omitting the unset ones.
        """
        found: dict[preason.Operation, str] = {}
        for operation, field in _OPERATION_FIELDS.items():
            value = getattr(self, field, None)
            if value:
                found[operation] = value  # type: ignore[index]
        return found

    def spec_for(self, name: str, *, drop_global_key: bool = False) -> ModelSpec:
        """
        Resolve one model's call options, per-model values over global ones.

        :param name: The model to resolve.
        :param drop_global_key: Whether to withhold the global ``api_key`` and
            ``api_base``. Set when a fallback chain is configured: a global key
            belongs to one provider and is wrong for every other model in the
            chain, and an authentication failure on a fallback is a confusing way
            to find that out.
        :return: The resolved spec.
        """
        options = self.models.get(name, ModelOptions())
        return ModelSpec(
            name=name,
            api_key=None if drop_global_key else self.api_key,
            api_base=options.api_base or (None if drop_global_key else self.api_base),
            temperature=(
                options.temperature if options.temperature is not None else self.temperature
            ),
            max_tokens=(options.max_tokens if options.max_tokens is not None else self.max_tokens),
            request_timeout=(
                options.request_timeout
                if options.request_timeout is not None
                else self.request_timeout
            ),
            reasoning_effort=(
                options.reasoning_effort
                if options.reasoning_effort is not None
                else self.reasoning_effort
            ),
        )

    def chain_for(self, primary: str | None) -> tuple[str, ...]:
        """
        Build the ordered list of models to try for one call.

        :param primary: The model routing chose, or ``None`` for the default.
        :return: The primary followed by the fallbacks, deduplicated, order kept.
        """
        ordered = [primary or self.model, *self.fallback_models]
        seen: set[str] = set()
        chain: list[str] = []
        for name in ordered:
            if name and name not in seen:
                seen.add(name)
                chain.append(name)
        return tuple(chain)


def _requested_config_file(
    init_settings: pydantic_settings.PydanticBaseSettingsSource,
) -> pathlib.Path | None:
    """
    Read the ``config_file`` a caller passed to the constructor.

    :param init_settings: The init-kwargs source.
    :return: The path, or ``None`` when none was given or the file is absent.
    """
    kwargs = getattr(init_settings, "init_kwargs", {}) or {}
    raw = kwargs.get("config_file")
    if raw is None:
        return None
    path = pathlib.Path(raw)
    return path if path.is_file() else None


def find_config_file(start: pathlib.Path | None = None) -> pathlib.Path | None:
    """
    Walk up from a directory looking for ``crux.toml``.

    Called by the CLI only. A library that did this would pick up whichever repo
    its host happened to be running in, which is the same category of ambient
    surprise as reading someone else's ``.env``.

    :param start: Where to start. Defaults to the working directory.
    :return: The file, or ``None`` when there is none above ``start``.
    """
    here = (start or pathlib.Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / "crux.toml"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: pathlib.Path | None = None) -> tuple[str, ...]:
    """
    Push a ``.env`` file into the process environment.

    pydantic-settings reads ``.env`` too, but only for ``CRUX_``-prefixed names.
    Provider credentials do not carry that prefix and litellm reads the real
    environment, so ``ANTHROPIC_API_KEY=...`` in a ``.env`` file would otherwise
    be silently ignored, which is the first thing anyone tries.

    Called from the CLI only. A library has no business mutating its host's
    environment; a host sets its own variables or passes :class:`CruxSettings`.

    Existing variables win: a shell export is a deliberate override of a file.

    :param path: The file to read. Defaults to ``.env`` in the working directory.
    :return: The names that were set, for logging.
    """
    target = path or pathlib.Path(".env")
    if not target.is_file():
        return ()
    loaded: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip()
        value = value.strip().strip("\"'")
        if name and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return tuple(loaded)
