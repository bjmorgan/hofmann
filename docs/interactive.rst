Interactive viewer
==================

The interactive viewer opens a matplotlib window where you can explore
a structure with mouse and keyboard controls.  When the window is closed
the adjusted :class:`~hofmann.ViewState` and :class:`~hofmann.RenderStyle`
are returned, ready for static rendering.

.. note::

   The interactive viewer requires a GUI-capable matplotlib backend
   such as **QtAgg**, **TkAgg**, or **macosx**.  Non-interactive
   backends (``Agg``, ``pdf``, ``svg``) will not display a window.

   In a Jupyter notebook, use the ``%matplotlib qt`` or
   ``%matplotlib tk`` magic before calling the viewer.  The default
   ``inline`` backend does not support interactive windows.

.. code-block:: python

   view, style = scene.render_mpl_interactive()

   # Reuse the adjusted view and style for static output:
   scene.view = view
   scene.render_mpl("output.svg", style=style)

You can pass an initial :class:`~hofmann.RenderStyle` or override
individual fields as keyword arguments:

.. code-block:: python

   view, style = scene.render_mpl_interactive(show_bonds=False)


Mouse controls
--------------

- **Left-drag** rotates the structure.
- **Scroll wheel** zooms in and out.


Keyboard controls
-----------------

Press **h** during the interactive session to show a help overlay
listing all keybindings.

Rotation and zoom
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Action
   * - Arrow keys
     - Rotate around horizontal / vertical axes
   * - ``,`` / ``.``
     - Roll (rotate in the screen plane)
   * - ``+`` or ``=``
     - Zoom in
   * - ``-``
     - Zoom out

Pan and perspective
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Action
   * - Shift + Arrow keys
     - Pan the view
   * - ``p`` / ``P``
     - Increase / decrease perspective strength
   * - ``d`` / ``D``
     - Increase / decrease viewing distance

Display toggles
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Action
   * - ``b``
     - Toggle bonds
   * - ``o``
     - Toggle outlines
   * - ``e``
     - Toggle polyhedra
   * - ``u``
     - Toggle unit cell edges
   * - ``a``
     - Toggle axes orientation widget

Frame navigation
^^^^^^^^^^^^^^^^

For scenes with multiple frames (trajectories):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Action
   * - ``[`` / ``]``
     - Step to previous / next frame
   * - ``{`` / ``}``
     - Jump to first / last frame

Other
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Action
   * - ``r``
     - Reset view to initial state
   * - ``h``
     - Toggle help overlay


Return values
-------------

The interactive viewer returns a ``(ViewState, RenderStyle)`` tuple.
Any changes made during the session — rotation, zoom, perspective,
and display toggles — are captured in the returned objects.

.. code-block:: python

   view, style = scene.render_mpl_interactive()

   # The view captures rotation, zoom, pan, perspective, and distance.
   scene.view = view

   # The style captures display toggles (bonds, outlines, polyhedra, etc.).
   scene.render_mpl("output.svg", style=style)

The returned :class:`~hofmann.RenderStyle` uses publication-quality
polygon counts (``circle_segments=72``, ``arc_segments=12``) even though
the interactive session uses lower-fidelity settings for responsiveness.
