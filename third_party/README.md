# Third Party

External source checkouts and vendored dependencies belong here, preferably as
submodules or documented external checkouts.

The current `_ct_cutlass/cutlass` path is an embedded git repository recorded by
the checkpoint commit as a gitlink. That is migration debt: either convert it to
a real submodule with a URL, or remove the checkout and keep only the local
kernel sources that this repository owns.
