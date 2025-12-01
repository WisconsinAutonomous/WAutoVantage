#include <Python.h>
#include <OpenGL/gl.h>
#include <cstring>

static PyObject* read_framebuffer(PyObject* self, PyObject* args) {
    int width, height;
    if (!PyArg_ParseTuple(args, "ii", &width, &height)) {
        return NULL;
    }
    
    int size = width * height * 3;
    int row_size = width * 3;
    
    // Read into temporary buffer
    char* temp = new char[size];
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, temp);
    
    // Create output buffer and flip vertically
    PyObject* bytes_obj = PyBytes_FromStringAndSize(NULL, size);
    if (!bytes_obj) {
        delete[] temp;
        return NULL;
    }
    
    char* buffer = PyBytes_AsString(bytes_obj);
    
    // Flip rows
    for (int y = 0; y < height; y++) {
        memcpy(buffer + y * row_size, 
               temp + (height - 1 - y) * row_size, 
               row_size);
    }
    
    delete[] temp;
    return bytes_obj;
}

static PyMethodDef methods[] = {
    {"read_framebuffer", read_framebuffer, METH_VARARGS, "Fast glReadPixels"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "fast_capture",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_fast_capture(void) {
    return PyModule_Create(&module);
}