import pyfiglet

def print_string(string):
    ascii_art = pyfiglet.figlet_format(string)
    print(ascii_art)


def print_header(header):
    border = '-' * (len(header) + 8)
    print(border)
    print(f"--- {header} ---")
    print(border)


def print_line():
    print('=' * 80)