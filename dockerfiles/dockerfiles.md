# How to use the docker files?

Build a Dockerfile for training into a Docker image:

- docker build -f train.dockerfile . -t train:latest

Mount a volume that is shared between the host(your laptop) and the container(-v option for the docker run command). This will allow us to get the the trained_model.pt from the container into a folder in our laptop. Run the image in a docker container:

- docker run --name {container_name} -v %cd%/models:/models/ train:latest

you can check the name of the docker container with:
 - docker ps -a

Also note that the %cd% needs to change depending on your OS. For linux and mac os $[PWD]: should work.


After training the model and saving it under /models, we build a new container for the evaluation:

- docker build -f evaluate.dockerfile . -t train:latest

we mount the volume and run the image in a docker container:

- docker run --name {container_name} -v %cd%/models:/models/ train:latest

Note that the above command should be adjusted with the paths of the files/folders to be saved from and to.

# Easier with invoke

Run docker build to achieve all these