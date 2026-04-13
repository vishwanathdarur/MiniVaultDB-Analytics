#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

#include "db/db.hpp"

namespace py = pybind11;

class PyMiniVaultDB {
public:
    PyMiniVaultDB(const std::string& path, size_t memtable_bytes = 64 * 1024 * 1024)
        : db_(std::make_unique<mvdb::DB>(path, memtable_bytes)) {}

    void put(const std::string& key, const std::string& value) {
        std::string key_mut = key;
        std::string value_mut = value;
        db_->put(key_mut, value_mut);
    }

    py::object get(const std::string& key) const {
        std::string key_mut = key;
        std::string value;
        if (db_->get(key_mut.data(), static_cast<uint32_t>(key_mut.size()), value)) {
            return py::cast(value);
        }
        return py::none();
    }

    void delete_key(const std::string& key) {
        std::string key_mut = key;
        db_->del(key_mut);
    }

    std::vector<std::pair<std::string, std::string>> scan() const {
        return db_->scan();
    }

private:
    std::unique_ptr<mvdb::DB> db_;
};

PYBIND11_MODULE(_native, m) {
    m.doc() = "MiniVaultDB C++ backend binding";

    py::class_<PyMiniVaultDB>(m, "MiniVaultDB")
        .def(py::init<const std::string&, size_t>(),
             py::arg("path"), py::arg("memtable_bytes") = 64 * 1024 * 1024)
        .def("put", &PyMiniVaultDB::put)
        .def("get", &PyMiniVaultDB::get)
        .def("delete", &PyMiniVaultDB::delete_key)
        .def("scan", &PyMiniVaultDB::scan);
}
