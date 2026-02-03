from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from src.peptide_designer import PeptideDesigner
from src.peptide_mutator import PeptideMutator
from src.filters.f_cytotox import CytotoxicityFilter
import src.utils.utils_visualization as visualizations
from src.generators.architectures.architectures import GenRNN
from src.config import GEN_PATH, CTT_PATH, DIST_DATA_PATH, OUTPUT_DIR, FOLDER_SIGNATURE, DROP_COLS, DEVICE
from src.utils.utils import get_next_run_id

import pandas as pd

app = Flask(__name__)
app.secret_key = "your-secret-key"

@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/generate', methods=['GET','POST'])
def generate():

    if request.method == 'POST':

        generator = request.form.get('model')
        device = request.form.get('device')
        n = int(request.form['n_sequences'])

        print('\n',generator,'\n')

        #print(generator)
        run_id = get_next_run_id(base_dir=OUTPUT_DIR / 'GENERATE')
        run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))

        # run pipeline
        designer = PeptideDesigner(
            gen_path=GEN_PATH / generator,
            f_ctt=CTT_PATH,
            f_dist=DIST_DATA_PATH,
            device=device
        )
        output_folder = OUTPUT_DIR / 'GENERATE' / run_name
        designer.run(
            n=n,
            output_dir=output_folder,
            drop_cols=DROP_COLS
        )

        visualizations = ['distribution_toxicity.png','latent_space.png',]

        return render_template('results.html',
                        run_name = run_name,
                        device=device,
                        output_dir=output_folder, 
                        visualizations = visualizations,
                        results_file='generated_sequences.csv'
        )

    return render_template('generate.html')

@app.route('/mutate', methods=['GET','POST'])
def mutate():

    if request.method == 'POST':
        
        target_sequence = request.form.get('sequence')
        selected_db = request.form.get('database')

        print(selected_db)

        run_id = get_next_run_id(base_dir=OUTPUT_DIR / 'MUTATE')
        run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))

        mutator = PeptideMutator(
            db=selected_db
        )
        output_folder = OUTPUT_DIR / 'MUTATE' / run_name
        mutator.run(
            target_sequence=target_sequence,
            output_dir=output_folder
        )

        visualizations = ['distribution_toxicity.png','distribution_charge.png']

        return render_template('results.html',
                run_name = run_name,
                output_dir=output_folder, 
                visualizations = visualizations,
                results_file='mutants.csv'
        )

    return render_template('mutate.html')

@app.route('/toxicity', methods=['GET','POST'])
def toxicity():
    if request.method == 'POST':

        if 'sequence_file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['sequence_file']
        device = request.form.get('device')

        try:
            # read directly from the file stream
            df = pd.read_csv(file)

            run_id = get_next_run_id(base_dir=OUTPUT_DIR / 'TOXICITY')
            run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))

            clf = CytotoxicityFilter(CTT_PATH, device)
            output_folder = OUTPUT_DIR / 'TOXICITY' / run_name

            df_filtered = clf.filter_sequences(df)

                    # save general results 
            output_folder.mkdir(parents=True, exist_ok=True)
            visualizations.probability_distribution(df['toxicity_prob'], output_folder, col='red', name='toxicity')
            df.to_csv(output_folder / 'predictions.csv',index=False)

            vis = ['distribution_toxicity.png']

            return render_template('results.html',
                    run_name = run_name,
                    output_dir=output_folder, 
                    visualizations = vis,
                    results_file='predictions.csv'
            )

        except Exception as e:
            return f"Error reading CSV: {str(e)}", 400

    return render_template('toxicity.html')


@app.route('/<path:filename>')
def download_file(filename):
    return send_from_directory('./', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=5000)