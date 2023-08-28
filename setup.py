from setuptools import find_packages, setup
from pathlib import Path
import re
import json
import yaml

name = "ren_utils"
version = "0.0"
ppkg = Path(Path(__file__).parent,name)

# Create and update __init__.py
p = Path(ppkg,"__init__.py")
s_ver = f"__version__='{version}'\n"
if not p.exists():
    with open(p,"x") as f:
        f.write(s_ver)
else:
    with open(p,"r") as f:
        s=  f.read()

    o = re.search("__version__[\w\W]*?\n",s)
    if o is not None:
        s =re.sub("__version__[\w\W]*?\n",s_ver,s)
    else:
        s+=f"\n{s_ver}"
    
    with open(p,"w") as f:
        f.write(s)

# Create apps/rennet.json
_d = {
    "root_Results":Path(ppkg.parent,"Results").as_posix(),
    "datasets":{
        "Dumb":{
            "imgs":"",
            "suffix":"jpg"
            }
        }
    }
p= Path(ppkg,"rennet.json")
if not p.exists():
    with open(p,"x") as f:
        f.write(json.dumps(_d))
else:
    print(f"- Already exists: {p}")

# Create mentioned folder 
Path(_d["root_Results"]).mkdir(parents=True,exist_ok=True)
Path(_d["pcache_invK"]).mkdir(parents=True,exist_ok=True)


# with open('requirements.yml') as f:
#     requirements = yaml.safe_load(f)
# 
# install_requires = []
# for package in requirements['dependencies']:
#     if isinstance(package, str):
#         install_requires.append(package)
#     elif isinstance(package, dict):
#         name = list(package.keys())[0]
#         version = package[name]
#         install_requires.append(f"{name}{version}")
        
setup(
    name=name,
    version=version,
    author='Yumeng REN',
    author_email='rymoon42@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[],
)