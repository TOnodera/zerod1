FROM jupyter/base-notebook

WORKDIR /workspaces/app
USER root
RUN apt-get update -y && apt-get install -y git vim
USER jovyan
RUN pip install numpy matplotlib
