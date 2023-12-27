__version__='0.0'

global_objects = {
    
}


def register_global_object(key,value):
    if key in global_objects:
        raise Exception(f"- RuntimeError: KeyAlreadyRegistered:{key}")
    global_objects[key]=  value
    
def get_global_object(key):
    return global_objects[key]

def has_global_object(key):
    return key in global_objects