#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSGeneratorImpl.h>

#include <torch/csrc/Generator.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_numbers.h>

using namespace torch;

static PyObject* MPSModule_initExtension(PyObject* self, PyObject* noargs) {
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.mps: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.mps module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS

  auto m = THPObjectPtr(PyImport_ImportModule("torch.mps"));
  if (!m)
    throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto gen = at::mps::detail::getDefaultMPSGenerator();
  auto default_mps_generator = (THPGenerator*)THPGenerator_initDefaultGenerator(gen);
  set_module_attr("default_mps_generator", (PyObject*) default_mps_generator);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_isAvailable(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::detail::getMPSHooks().hasMPS()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_synchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().deviceSynchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static at::mps::IMPSAllocator* getMPSAllocator() {
  auto allocator = static_cast<at::mps::IMPSAllocator*>(at::detail::getMPSHooks().getMPSDeviceAllocator());
  THPUtils_assert(allocator, "failed to get MPSAllocator interface");
  return allocator;
}

PyObject* MPSModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  getMPSAllocator()->emptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkDouble(args), "invalid argument to setMemoryFraction()");
  double highWatermarkRatio = THPUtils_unpackDouble(args);
  getMPSAllocator()->setHighWatermarkRatio(highWatermarkRatio);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* MPSModule_currentAllocatedMemory(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLongLong(getMPSAllocator()->getCurrentAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

PyObject* MPSModule_driverAllocatedMemory(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyLong_FromUnsignedLongLong(getMPSAllocator()->getDriverAllocatedMemory());
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,
// cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-avoid-c-arrays)
static struct PyMethodDef _MPSModule_methods[] = {
    {"_mps_init", MPSModule_initExtension, METH_NOARGS, nullptr},
    {"_mps_synchronize", MPSModule_synchronize, METH_NOARGS, nullptr},
    {"_is_mps_available", MPSModule_isAvailable, METH_NOARGS, nullptr},
    {"_mps_emptyCache", MPSModule_emptyCache, METH_NOARGS, nullptr},
    {"_mps_setMemoryFraction", MPSModule_setMemoryFraction, METH_O, nullptr},
    {"_mps_currentAllocatedMemory", MPSModule_currentAllocatedMemory, METH_NOARGS, nullptr},
    {"_mps_driverAllocatedMemory", MPSModule_driverAllocatedMemory, METH_NOARGS, nullptr},
};

PyMethodDef* MPSModule_methods() {
  return _MPSModule_methods;
}