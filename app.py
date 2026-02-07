from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session
from authlib.integrations.flask_client import OAuth
from dotenv import load_dotenv

from src.peptide_designer import PeptideDesigner
from src.peptide_mutator import PeptideMutator
from src.filters.f_cytotox import CytotoxicityFilter
import src.utils.utils_visualization as visualizations
from src.generators.architectures.architectures import GenRNN
from src.config import GEN_PATH, CTT_PATH, DIST_DATA_PATH, OUTPUT_DIR, FOLDER_SIGNATURE, DROP_COLS, DEVICE
from src.utils.utils import get_next_run_id

import pandas as pd
import os

from __version__ import __version__

app = Flask(__name__)
app.secret_key = "your-secret-key"

@app.context_processor
def inject_version():
    return dict(app_version=app.config['APP_VERSION'])

app.config['APP_VERSION'] = __version__

### --authentication-- ###
# oauth = OAuth(app)

# muni = oauth.register(
#     name='muni',
#     client_id=os.getenv('MUNI_CLIENT_ID'),          # ← from registration
#     client_secret=os.getenv('MUNI_CLIENT_SECRET'),  # ← from registration
#     server_metadata_url='https://login.muni.cz/oidc/.well-known/openid-configuration',  # confirm exact URL!
#     client_kwargs={
#         'scope': 'openid profile email',            # add 'offline_access' if you want refresh tokens
#         'prompt': 'select_account',                 # optional: force account selection
#     }
# )

@app.route('/')
def intro():
    # user = session.get('user')
    # if user:
    #     return f"""
    #     <h1>Welcome, {user.get('name', 'MUNI user')}!</h1>
    #     <p>Email: {user.get('email')}</p>
    #     <p>UČO / sub: {user.get('sub')}</p>
    #     <a href="/logout">Logout</a>
    #     """
    return render_template('intro.html')

# @app.route('/login')
# def login():
#     redirect_uri = url_for('auth_callback', _external=True)
#     return muni.authorize_redirect(redirect_uri)

# @app.route('/auth/callback')
# def auth_callback():
#     token = muni.authorize_access_token()
    
#     # Get user info (automatic via OIDC userinfo endpoint)
#     userinfo = muni.userinfo(token=token)
    
#     # Store in session (you can also save to DB + use flask-login)
#     session['user'] = userinfo
#     session['token'] = token  # access_token, id_token, refresh_token...

#     return redirect(url_for('index'))


### --logic-- ###

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
        selected_db_file = request.files.get('DB_file')

        if selected_db_file is None or selected_db_file.filename == '':
            selected_db = request.form.get('database')
        else:
            print('DB uploaded')
            selected_db_file = request.files['DB_file']
            selected_db = pd.read_csv(selected_db_file)

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
                run_name=run_name,
                output_dir=output_folder, 
                visualizations=visualizations,
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