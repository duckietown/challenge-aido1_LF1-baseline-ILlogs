define-challenge:
	$(MAKE) define-challenge-LF
	$(MAKE) define-challenge-LFV
	$(MAKE) define-challenge-LFVI

define-challenge-no-cache:
	$(MAKE) define-challenge-LF-no-cache
	$(MAKE) define-challenge-LFV-no-cache
	$(MAKE) define-challenge-LFVI-no-cache

define-challenge-LF:
	dts challenges define --config LF.challenge.yaml

define-challenge-LFV:
	dts challenges define --config LFV.challenge.yaml

define-challenge-LFVI:
	dts challenges define --config LFVI.challenge.yaml

define-challenge-LF-no-cache:
	dts challenges define --config LF.challenge.yaml  --no-cache

define-challenge-LFV-no-cache:
	dts challenges define --config LFV.challenge.yaml  --no-cache

define-challenge-LFVI-no-cache:
	dts challenges define --config LFVI.challenge.yaml  --no-cache


test-with-local-repos:
	docker-compose -f docker-compose-devel.yaml down -v
	#docker-compose -f docker-compose-devel.yaml build
	docker-compose -f docker-compose-devel.yaml up -V --build

test-regular:
	docker-compose -f docker-compose.yaml down -v
	docker-compose -f docker-compose.yaml build --pull
	docker-compose -f docker-compose.yaml up -V --build


test-regular-no-cache:
	docker-compose -f docker-compose.yaml down -v
	docker-compose -f docker-compose.yaml build --pull --no-cache
	docker-compose -f docker-compose.yaml up -V --build
