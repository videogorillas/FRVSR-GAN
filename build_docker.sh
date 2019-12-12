#!/bin/bash -xue

LABEL=$1

# docker login r.c.videogorillas.com
HOST=r.c.videogorillas.com
#HOST=kote.local:31337
docker build -t $HOST/up4k:$LABEL .
docker push $HOST/up4k:$LABEL
