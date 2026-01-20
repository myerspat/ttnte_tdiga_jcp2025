# Variables
ZIP_NAME = ttnte_tdiga_jcp2025_data.tar.gz
SOURCES = \
					fixed_source/square/direction/direction.jsonl \
					fixed_source/square/direction/processed_direction.jsonl \
					fixed_source/square/direction/meshes \
					fixed_source/square/direction/rnorms \
					fixed_source/square/meshsize/meshsize.jsonl \
					fixed_source/square/meshsize/processed_meshsize.jsonl \
					fixed_source/square/meshsize/meshes \
					fixed_source/square/meshsize/rnorms \
					fixed_source/circle/direction/direction.jsonl \
					fixed_source/circle/direction/processed_direction.jsonl \
					fixed_source/circle/direction/meshes \
					fixed_source/circle/direction/rnorms \
					fixed_source/circle/meshsize/meshsize.jsonl \
					fixed_source/circle/meshsize/processed_meshsize.jsonl \
					fixed_source/circle/meshsize/meshes \
					fixed_source/circle/meshsize/rnorms \
					fixed_source/quarter_circle/direction/direction.jsonl \
					fixed_source/quarter_circle/direction/processed_direction.jsonl \
					fixed_source/quarter_circle/direction/meshes \
					fixed_source/quarter_circle/direction/rnorms \
					fixed_source/quarter_circle/meshsize/meshsize.jsonl \
					fixed_source/quarter_circle/meshsize/processed_meshsize.jsonl \
					fixed_source/quarter_circle/meshsize/meshes \
					fixed_source/quarter_circle/meshsize/rnorms \
					fixed_source/cruciform/data.pkl \
					eigenvalue/circle/stats.pkl \
					eigenvalue/circle/solutions.pkl \
					eigenvalue/quarter_circle/data.pkl \
					eigenvalue/pincell/data.pkl \
					eigenvalue/lightbridge_ba/data.pkl \
					eigenvalue/lightbridge_gas/stats.pkl \
					eigenvalue/lightbridge_gas/solutions.pkl

zip:
	@echo "Creating tarball $(ZIP_NAME)"
	tar -cvzf $(ZIP_NAME) $(SOURCES)

unzip:
	@echo "Extracting $(ZIP_NAME)"
	tar -xvzf $(ZIP_NAME)

clean:
	rm $(ZIP_NAME)

.PHONY: zip unzip clean
