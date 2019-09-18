FROM kbase/sdkbase2:python
# FROM kbase/sdkbase2:python #try this again

# FROM kbase/kb_sdk_docs:latest
# MAINTAINER KBase Developer

# -----------------------------------------
# In this section, you can install any system dependencies required
# to run your App.  For instance, you could place an apt-get update or
# install line here, a git checkout to download code, or run any other
# installation scripts.

# RUN apt-get upgrade -y
RUN apt-get update

RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install seaborn
RUN python -mpip install matplotlib
RUN pip install graphviz

RUN pip install xlrd

RUN pip install --upgrade pip

RUN pip install coverage

RUN pip install xlsxwriter

RUN apt-get install python-tk -y
ENV DISPLAY :0

# -----------------------------------------

RUN apt-get install graphviz -y
	
COPY ./ /kb/module
RUN mkdir -p /kb/module/work
RUN chmod -R a+rw /kb/module


WORKDIR /kb/module


RUN make all

ENTRYPOINT [ "./scripts/entrypoint.sh" ]

CMD [ ]
