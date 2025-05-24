import tensorflow as tf
import sys

def check_gpu():
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("\nGPU Devices:")
    print(tf.config.list_physical_devices('GPU'))
    
    # Test GPU memory
    try:
        gpu = tf.config.experimental.get_visible_devices('GPU')[0]
        print("\nGPU Memory:")
        print(tf.config.experimental.get_memory_info('GPU:0'))
    except:
        print("\nNo GPU found or error accessing GPU memory")

if __name__ == "__main__":
    check_gpu() 