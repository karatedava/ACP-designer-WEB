from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from src.acp_maker import ACPmaker
from src.generators.architectures.architectures import GenRNN
from src.config import GENGRU_PATH, CTT_PATH, DIST_DATA_PATH, OUTPUT_DIR, FOLDER_SIGNATURE, DROP_COLS, DEVICE

app = Flask(__name__)
app.secret_key = "your-secret-key"

@app.route('/', methods=['GET','POST'])
def intro():

    if request.method == 'POST':

        device = request.form.get('device')
        nbatches = int(request.form['nbatches'])

        run_id = 420
        run_name = FOLDER_SIGNATURE.replace('XX',str(run_id))

        # run pipeline
        acp_maker = ACPmaker(GENGRU_PATH, CTT_PATH, DIST_DATA_PATH)
        output_folder = OUTPUT_DIR / run_name
        acp_maker.run_pipeline(nbatches, output_folder, DROP_COLS ,None)

        visualizations = ['distribution.png','latent_space.png',]

        return render_template('results.html',
                        run_name = run_name,
                        device=device,
                        output_dir=output_folder, 
                        visualizations = visualizations,
                        results_file='results_all.csv'
        )


    return render_template('intro.html')

if __name__ == '__main__':
    app.run(debug=False)