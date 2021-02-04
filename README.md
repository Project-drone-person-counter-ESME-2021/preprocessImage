# preprocessImage
create one image from multiple image

## Create venv
linux: 

to know where is your python2.7, go to racine folder :
```
which python2.7
```
if you have no python2.7, install it : <br />
https://linuxconfig.org/install-python-2-on-ubuntu-20-04-focal-fossa-linux

Now, you can create a new virtual environment with the command :

```bash
virtualvenv -p where_is_python/python2.7 name_venv
```
tips: there no git ignore, do not create it in the folder of the git and push it. <br />

## Use the virtual environment
To activate it, use this command :
```bash
source where_is_virtual_environment/bin/activate
```

you should see the name of the virtual environment in the terminal
if not, it could be a problem
to check, you could use the command :
```bash
python -V
```
You should see that the version is 2.7

## install the lib
```bash
pip install -r root_git_project/requirements.txt
```

## launch a python file
to launch a python file, use this command :
```bash
python the_file.py 
```

concat_img.py : Is for getting one image from multiple image
create_multiple_image_from_panorama.py : Is to create image test
index.py : Is just a test

## final :
Use propre.py :
To do all thing one time use concat function, it takes left and right image and return concat image of the two.









