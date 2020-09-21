install:
	conda env create -f ./environment.yml && conda activate px
	cd ui && npm install && npm run build

update:
	conda env update
	cd ui && npm install && npm run build

build:
	cd ui && npm run build

download-example-autoencoders:
	cd examples && python download-autoencoders.py

example-3kb:
	cd examples && python download-example-3kb.py && cd .. && ./start.py -d -c examples/config-example-3kb.json

example-12kb:
	cd examples && python download-example-12kb.py && cd .. && ./start.py -d -c examples/config-example-12kb.json

example-120kb:
	cd examples && python download-example-120kb.py && cd .. && ./start.py -d -c examples/config-example-120kb.json

encode-e11-5-limb:
	cd examples && python download-encode-e11-5-limb.py && cd .. && ./start.py -d -c examples/config-encode-e11-5-limb.json

encode-e11-5-face-hindbrain:
	cd examples && python download-encode-e11-5-face-hindbrain.py && cd .. && ./start.py -d -c examples/config-encode-e11-5-face-hindbrain.json

roadmap-e116-gm12878:
	cd examples && python download-roadmap-e116-gm12878.py && cd .. && ./start.py -d -c examples/config-roadmap-e116-gm12878.json
