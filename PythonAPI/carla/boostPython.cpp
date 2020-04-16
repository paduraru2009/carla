// boostPython.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <boost/python.hpp>
using namespace boost::python;


struct World
{
    void set(std::string msg) { this->msg = msg; }
    std::string greet() { return msg; }
    std::string msg;
};
/*
void PyInit_libcarla(void)
{

}

PyMODINIT_FUNC PyInit__libcarla(void)
{
	
}*/

#if PY_VERSION_HEX >= 0x03000000

#else
UN MARE KKT
#endif




BOOST_PYTHON_MODULE(libcarla)
{
    class_<World>("libcarla", init<>())
        .def("greet", &World::greet)
        .def("set", &World::set);
}
