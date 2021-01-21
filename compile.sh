# C++ executable
g++ -Wall -O2 ./HS.cpp -o HS_cpp

# C++ python extension
# Need to install pybind11 package
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) heuristic.cpp -o heuristic$(python3-config --extension-suffix)
