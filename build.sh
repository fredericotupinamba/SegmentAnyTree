#/bin/sh
# Builds the patched image (see Dockerfile.pandas-fix) and tags it for
# push to Docker Hub under fredericotupinamba/tupisat. Push with:
#   docker login -u fredericotupinamba
#   docker push fredericotupinamba/tupisat:latest
docker build -f Dockerfile.pandas-fix -t fredericotupinamba/tupisat:latest .
