install:
	conda env create -f ./environment.yml
	cd ui && npm install && cd node_modules/higlass && npm install && npm run build

update:
	conda env update
	cd ui && rm -rf node_modules/higlass && npm install && cd node_modules/higlass && npm install && npm run build

build:
	cd ui && npm run build

example-2_4kb:
	cd examples && ./prepare-2_4kb.py && cd .. && ./start.py --debug --config=examples/config-dnase-seq-2_4kb.json

example-12kb:
	cd examples && ./prepare-12kb.py && cd .. && ./start.py --debug --config=examples/config-chip-seq-12kb.json
