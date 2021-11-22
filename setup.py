
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
REQUIREMENTS = ["numpy", "torch", "opencv-python", "trimesh", "vedo"]

setuptools.setup(
     name='na',  
     version='0.0.1',
     #scripts=['dokr'],
     author=["Yansel González Tejeda", "Helmut A. Mayer"],
     author_email="neoglez@gmail.com",
     install_requires=REQUIREMENTS,
     description="A Neural Anthropometer Learning from Body Dimensions Computed on Human 3D Meshes",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/neoglez/neural-antropometer",
     packages=setuptools.find_packages(
         exclude=[
             "datageneration",
             "dataset",
             "img",
             "model",
             "notebook",
             "results",
             "scenes",
             ".vscode"]),
     classifiers=[
         "Programming Language :: Python :: 2",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],

 )
