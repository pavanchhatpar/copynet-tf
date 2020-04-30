FROM tensorflow/tensorflow:latest-gpu-py3-jupyter AS nb
WORKDIR /tf/src
COPY . /tf/src
ARG JUPYTER_PASSWD
ENV HOME /tf
RUN pip uninstall -y enum34\
    && pip install --upgrade pip setuptools\
    && pip install -r requirements.txt\
    && python -m spacy download en_core_web_sm\
    && jupyter notebook --generate-config\
    && mv /root/.jupyter /.jupyter\
    && v="from notebook.auth import passwd; print(passwd('`echo $JUPYTER_PASSWD`'))"\
    && python -c "$v" > tmp.txt\
    && echo "c.NotebookApp.password='`cat tmp.txt`'" >> /.jupyter/jupyter_notebook_config.py\
    && cp -r /.jupyter /tf\
    && chmod -R 777 /tf/.jupyter

FROM nb AS bash
CMD ["bash"]