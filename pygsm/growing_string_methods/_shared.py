def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj, methname)(*arg[2:])