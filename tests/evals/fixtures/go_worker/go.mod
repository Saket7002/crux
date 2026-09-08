// Synthetic fixture. crux's filesystem retriever reads this to answer "what
// does this project already depend on". The versions are deliberately not real
// releases, so GitHub's dependency graph has nothing to match: real ones
// produced permanent false-positive alerts against a project that does not
// exist and is never built.
module example.com/worker-fixture

go 1.22

require (
	github.com/redis/go-redis/v9 v0.0.0-fixture
	go.uber.org/zap v0.0.0-fixture
)
