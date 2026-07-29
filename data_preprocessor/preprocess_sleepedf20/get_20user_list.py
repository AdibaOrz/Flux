"""
Based on this code https://gist.github.com/emadeldeen24/a22691e36759934e53984289a94cb09b
"""

DOWNLOAD_STRING = """
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4001E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4001E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4001EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4002E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4002E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4002EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4011E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4011E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4011EH-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4012E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4012E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4012EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4021E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4021E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4021EH-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4022E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4022E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4022EJ-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4031E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4031E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4031EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4032E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4032E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4032EP-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4041E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4041E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4041EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4042E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4042E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4042EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4051E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4051E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4051EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4052E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4052E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4052EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4061E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4061E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4061EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4062E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4062E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4062EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4071E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4071E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4071EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4072E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4072E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4072EH-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4081E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4081E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4081EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4082E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4082E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4082EP-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4091E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4091E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4091EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4092E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4092E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4092EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4101E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4101E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4101EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4102E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4102E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4102EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4111E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4111E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4111EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4112E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4112E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4112EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4121E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4121E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4121EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4122E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4122E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4122EV-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4131E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4131E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4131EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4141E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4141E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4141EU-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4142E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4142E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4142EU-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4151E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4151E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4151EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4152E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4152E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4152EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4161E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4161E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4161EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4162E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4162E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4162EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4171E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4171E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4171EU-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4172E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4172E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4172EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4181E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4181E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4181EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4182E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4182E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4182EC-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4191E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4191E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4191EP-Hypnogram.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4192E0-PSG.edf
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4192E0-PSG.edf.hyp
wget https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/SC4192EV-Hypnogram.edf
"""

def extract_filenames():
    filenames = []
    for line in DOWNLOAD_STRING.split("\n"):
        if line.startswith("wget"):
            filename = line.split('/')[-1].split(".")[0].split("-")[0][3:5]
            filenames.append(filename)
    filenames = list(set(filenames))
    filenames.sort()
    return filenames


def main():
    filenames = extract_filenames()
    for idx, filename in enumerate(filenames):
        print(f"[{idx}]: {filename}")

    print(f"user list:\n{filenames}")


# OUTPUT: ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
if __name__ == "__main__":
    main()