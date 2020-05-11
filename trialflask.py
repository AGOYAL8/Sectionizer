

from flask import Flask, flash, redirect, render_template, request, session, abort
import os

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/upload_docx', methods=['GET','POST'])
def upload():
    if request.method == "POST":
        if request.files:
            document = request.files["document"]
            # document = os.path.join(APP_ROOT, 'docs/')
            print(document)
            # return redirect(request.url)
    # file = request.files['inputFile']
    return render_template("upload_new_docx.html", filename=document)
    # return app.send_static_file("upload-docx.html")
    # return file.filename


if __name__ == '__main__':
    app.run(debug=True)

