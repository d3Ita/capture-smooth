"""List DirectShow video devices in the same order OpenCV (CAP_DSHOW) numbers them."""
from ctypes import POINTER, c_ulong, c_void_p

import comtypes
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown, STDMETHOD
from comtypes.client import CreateObject
from comtypes.persist import IPropertyBag

CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")


class IMoniker(IUnknown):
    _iid_ = GUID("{0000000f-0000-0000-C000-000000000046}")


class IEnumMoniker(IUnknown):
    _iid_ = GUID("{00000102-0000-0000-C000-000000000046}")


class ICreateDevEnum(IUnknown):
    _iid_ = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")


# Only BindToStorage is used; the methods before it keep the vtable slots right.
IMoniker._methods_ = [
    STDMETHOD(HRESULT, "GetClassID", [c_void_p]),
    STDMETHOD(HRESULT, "IsDirty", []),
    STDMETHOD(HRESULT, "Load", [c_void_p]),
    STDMETHOD(HRESULT, "Save", [c_void_p, c_ulong]),
    STDMETHOD(HRESULT, "GetSizeMax", [c_void_p]),
    STDMETHOD(HRESULT, "BindToObject", [c_void_p, c_void_p, c_void_p, c_void_p]),
    COMMETHOD([], HRESULT, "BindToStorage",
              (["in"], c_void_p, "pbc"),
              (["in"], c_void_p, "pmkToLeft"),
              (["in"], POINTER(GUID), "riid"),
              (["out"], POINTER(POINTER(IUnknown)), "ppvObj")),
]
IEnumMoniker._methods_ = [
    COMMETHOD([], HRESULT, "Next",
              (["in"], c_ulong, "celt"),
              (["out"], POINTER(POINTER(IMoniker)), "rgelt"),
              (["out"], POINTER(c_ulong), "pceltFetched")),
]
ICreateDevEnum._methods_ = [
    COMMETHOD([], HRESULT, "CreateClassEnumerator",
              (["in"], POINTER(GUID), "clsidDeviceClass"),
              (["out"], POINTER(POINTER(IEnumMoniker)), "ppEnumMoniker"),
              (["in"], c_ulong, "dwFlags")),
]


def video_devices():
    """Return the DirectShow video device names; index i is OpenCV's device i."""
    comtypes.CoInitialize()
    dev_enum = CreateObject(CLSID_SystemDeviceEnum, interface=ICreateDevEnum)
    enum = dev_enum.CreateClassEnumerator(CLSID_VideoInputDeviceCategory, 0)
    names = []
    if not enum:
        return names
    while True:
        moniker, fetched = enum.Next(1)
        if not fetched:
            break
        bag = moniker.BindToStorage(None, None, IPropertyBag._iid_).QueryInterface(IPropertyBag)
        names.append(str(bag.Read("FriendlyName", pErrorLog=None)))
    return names


if __name__ == "__main__":
    for i, n in enumerate(video_devices()):
        print(i, n)
