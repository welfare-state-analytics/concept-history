import subprocess
import yaml
import os

def populate(s, params):
    for key in params:
        m = key + "_PLACEHOLDER"
        s = s.replace(m, str(params[key]))
    return s

params = {
        'corpus':'../riksdagen-corpus/corpus',
        'target':'data/target-words.json',
        'stopwords':'data/stopwords-sv.txt',
        'data':'data/context_windows',
        'results':'results/bash-test-run',

        'epochs':2,
        'burn_in':1,
        'sample_intervals':1,
        'alpha':0.5,
        'beta':0.5,

        'metadata': {
            'party_abbrev',
            'gender',
            'born',
            'id',
            },

        'window_sizes':None,
        'projects':None,
        'K':None,
        }

params = dict(corpus='../riksdagen-corpus/corpus',
    target='data/target-words.json',
    stopwords='data/stopwords-sv.txt',
    data='data/context_windows',
    results='results/bash-test-run',
    window_sizes=None,
    projects=None,
    epochs=2,
    burn_in=1,
    sample_intervals=1,
    alpha=0.5,
    beta=0.5,
    K=None,
    metadata=['party_abbrev', 'gender', 'born', 'id',],
    )

def individual_run(c, p, K):
    # To fix:
    # .sh script does not populate properly
    params["projects"] = [p]
    params["window_sizes"] = [c]
    params["K"] = [K]
    params_str_yaml = yaml.dump(params, default_flow_style=False)

    # Replace placeholders with actual parameter values
    s = open("bash-scripts/template.sh").read()

    params_str = "_".join([str(p), 'window', str(c), 'topic', str(K)])
    run_filename = 'bash-scripts/' + params["results"].split("/")[1] + "/run_" + params_str

    f_out = open(run_filename + ".sh", "w")
    f_out.write(s.format(run_filename + ".yml"))
    f_out.close()

    f_out = open(run_filename + ".yml", "w")
    f_out.write(params_str_yaml)
    f_out.close()

    # subprocess.call(['bash', run_filename + '.sh'])
    subprocess.call(['sbatch', run_filename + '.sh']) # something like this for cluster run

    os.remove(run_filename + ".sh")
    os.remove(run_filename + ".yml")

def main():
    try:
        os.mkdir(f'bash-scripts/{params["results"].split("/")[1]}')
    except:
        pass
    for c in [1, 2]:
        for p in ['fluga', 'skarv']:
            for K in [1, 2]:
                individual_run(c,p,K)

if __name__ == '__main__':
    main()
