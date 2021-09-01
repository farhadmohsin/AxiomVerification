import csv
from generation import *
from ties import generate_prefs

def batch_setup(batch_num, m, nmin, nmax, ind):
    os.chdir(config.data_folder)
    if str(batch_num) not in glob.glob("*"):
        os.mkdir(str(batch_num))
    os.chdir(str(batch_num))
    if "results" not in glob.glob("*"):
        os.mkdir("results")

    if "sat_results" not in glob.glob("*"):
        os.mkdir("sat_results")

    if "setup.json" not in glob.glob("*"):
        setup = dict()
        setup["m"] = m
        setup["nmin"] = nmin
        setup["nmax"] = nmax
        setup["ind"] = ind
        setup["generated"] = 0
        setup["computed"] = 0
        outfile = open("setup.json", 'w')
        json.dump(setup, outfile)
        outfile.close()
    infoname = str(batch_num) + "M=" + str(m) + "_start=" + str(nmin) + "_end=" + str(nmax) + "_inter = "+str(ind)+".txt"
    f = open(infoname, "w+")
    f.close()
if __name__ == '__main__':
    batch_num = int(sys.argv[1])
    m = int(sys.argv[2])
    nmin = int(sys.argv[3])
    nmax = int(sys.argv[4])
    ind = int(sys.argv[5])

    batch_setup(batch_num, m, nmin, nmax, ind)
    #generate_prefs(10, 50000)
    generate_prefs(batch_num, 150000)