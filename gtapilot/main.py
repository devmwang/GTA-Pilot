from multiprocessing import freeze_support

from coordinator.coordinator import main

# Launch coordinator
if __name__ == "__main__":
    freeze_support()
    main()
