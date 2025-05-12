#!/bin/bash
apt-get update
apt-get install -y swig zlib1g-dev libjpeg-dev
pip install -r requirements.txt 