// Synthetic fixture. crux's filesystem retriever reads this to answer "what
// does this project already depend on". The versions sit above every advisory
// range on purpose: real ones raised false-positive alerts against a project
// that does not exist, and 0.0.0 was worse, because advisory ranges are open at
// the bottom and matched it too.
module example.com/worker-fixture

go 1.22

require (
	github.com/redis/go-redis/v9 v99.0.0-fixture
	go.uber.org/zap v99.0.0-fixture
)
