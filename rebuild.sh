#!/bin/sh
docker-compose stop embed-nordic
docker-compose rm -f embed-nordic
docker-compose build embed-nordic
docker-compose up -d embed-nordic