# Development

## Testing

To run everything, use:

    $ make test-regular


The output will be in the directory `./challenges-dir/`.


## Testing using local repositories

You can also run the containers by mounting the local repositories.

To do so you must have the structure in [dt-env-developer](https://github.com/duckietown/dt-env-developer)
and set the variable `DT_ENV_DEVELOPER` to that repo's root directory.

To run use:

    $ make test-with-local-repos

You can see how the repos are mounted in the file [`docker-compose-devel.yaml`](docker-compose-devel.yaml).
