import tracemalloc
import contextlib
import time

from pydantic import BaseModel

from src.util.custom_logging import console


class Tracker(BaseModel):
    max_memory: int = 0
    time_taken: int = 0

    @property
    def max_memory_mb(self):
        return self.max_memory / 1024 / 1024
    
    @property
    def time_taken_seconds(self):
        return self.time_taken
    

@contextlib.contextmanager
def max_memory_tracker(tracker: Tracker):
    tracemalloc.start()
    yield
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracker.max_memory = peak
    console.print(f"Max memory usage: {peak / 1024 / 1024} MB")
    
    

@contextlib.contextmanager
def time_tracker(tracker: Tracker):
    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time
    tracker.time_taken = execution_time
    console.print(f"Execution time: {tracker.time_taken_seconds:.2f} seconds")
    


if __name__ == "__main__":

    tracker = Tracker(max_memory=0, time_taken=0)

    def my_function(n):

        a = [1] * n
        time.sleep(2)
    
    with max_memory_tracker(tracker=tracker), time_tracker(tracker=tracker):
        my_function(1000000)
        
    print(tracker.max_memory_mb, tracker.time_taken_seconds)