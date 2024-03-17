# Polish Natural Language Question Answering with limited context

This is a project to test RAGs for polish language.

## Getting started

Follow these steps:

### Create a virtual environment 

Create a Python 3.12 environment for this project and activate it:

(Replace with `3.12`, if you need)

```bash
python3.12 -m venv env
```

Activate it with:

```bash
source env/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

You can also use `Makefile` to setup everything for you:

```bash
make
source env/bin/activate
```

This will install all required dependencies and make the project and tools ready to use.
It will also instal the `mlx` package which is optional, but can be used on MacOS for better performance.

### Installing ElasticSearch and Qdrant

To start working with ES and QDrant install `docker/docker-composer` on your system and then run the following command:
```bash
docker-compose up -d
```

or 

```bash
make up
```