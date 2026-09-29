"""Building PDB records for tests, in the columns the format actually specifies.

Getting this wrong is easy and quiet: an atom name is 13-16, then column 17 is
altLoc, and dropping it shifts the residue name and the chain one place left, where a
reader finds a residue called " AL" on chain " ". Written out column by column so the
arithmetic is visible.
"""

BACKBONE = (" N  ", " CA ", " C  ", " O  ")
SIDECHAIN = (" CB ", " CG ", " CD ")


def atom(serial, name, residue, chain, resseq, xyz, b=50.0):
    x, y, z = xyz
    return (
        "ATOM  "               # 1-6    record
        f"{serial:>5d}"        # 7-11   serial
        " "                    # 12
        f"{name:<4s}"          # 13-16  atom name
        " "                    # 17     altLoc
        f"{residue:>3s}"       # 18-20  residue name
        " "                    # 21
        f"{chain}"             # 22     chain
        f"{resseq:>4d}"        # 23-26  residue number
        "    "                 # 27-30  insertion code and padding
        f"{x:8.3f}{y:8.3f}{z:8.3f}"   # 31-54
        "  1.00"               # 55-60  occupancy
        f"{b:6.2f}"            # 61-66  B factor, where pLDDT is written
        "\n"
    )


def residue(serial, residue_name, chain, resseq, origin, b=50.0, sidechain=0):
    """One residue: its backbone, and `sidechain` atoms beyond it."""
    x, y, z = origin
    names = BACKBONE + SIDECHAIN[:sidechain]
    return [atom(serial + i, name, residue_name, chain, resseq, (x + i, y, z), b)
            for i, name in enumerate(names)]


def chain_lines(chain, sequence, start=(0.0, 0.0, 0.0), b=50.0, sidechain=0):
    """A chain of residues, four atoms each plus `sidechain` more."""
    lines, serial = [], 1
    for index, name in enumerate(sequence):
        lines += residue(serial, name, chain, index + 1,
                         (start[0], start[1] + index * 4.0, start[2]), b, sidechain)
        serial += 4 + sidechain
    return lines
