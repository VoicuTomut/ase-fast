"""
Image scraper
=============

Dummy example to custom scrape image files.
"""


from ase.build import bulk
from ase.io import write

atoms = bulk('Si')
renderer = write('scraper.Si.pov', atoms)
renderer.render()
