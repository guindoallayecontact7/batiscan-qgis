def classFactory(iface):
    from .plugin import BatiScanPlugin
    return BatiScanPlugin(iface)
