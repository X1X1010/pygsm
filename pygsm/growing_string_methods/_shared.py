"""Helpers shared by the GSM modules."""

import multiprocessing as mp


def worker(arg):
    """Call ``getattr(obj, name)(*args)`` for ``arg = (obj, name, *args)``; picklable for ``Pool.map``."""
    obj, methname = arg[:2]
    return getattr(obj, methname)(*arg[2:])


def map_method(calls, mp_cores):
    """Evaluate each ``(obj, method_name, *args)`` call, in a process pool when ``mp_cores > 1``.

    Results are returned in input order. Objects are pickled to the workers, so
    any state the method changes on them is not reflected in the parent.
    """
    if mp_cores > 1:
        pool = mp.Pool(mp_cores)
        try:
            return pool.map(worker, calls)
        finally:
            pool.close()
            pool.join()
    return [worker(call) for call in calls]
