import resume_customizer_backend as bs
from flask import Flask, render_template, request, redirect, url_for, make_response
import pandas as pd
from io import BytesIO
import plotly.express as px
from urllib.parse import quote_plus, unquote_plus
import pdfkit


def plot_df(y, width, df, plot_title, xl, yl, figsz):
    fig = px.bar(
        df, 
        x=width, 
        y=y, 
        orientation='h'
    )
    fig.update_xaxes(title_text=xl)
    fig.update_yaxes(title_text=yl)
    # Convert the plot to HTML
    plot_html = fig.to_html()
    return plot_html

def generate_pdf(data, template):
    rendered_html = render_template(template, **data)
    pdf = pdfkit.from_string(rendered_html, False)
    return pdf

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        job_desc = request.form['job_desc']
        encoded_job_desc = quote_plus(job_desc)
        return redirect(url_for('analyze_jd', jd_desc=encoded_job_desc))
    return render_template('index.html')

@app.route('/analyze_jd/<jd_desc>')
def analyze_jd(jd_desc):
    jd_desc = unquote_plus(jd_desc)
    tfidf_df, word_freq_scores, entity_df, topic_df, skills, domain_based_on_skill_analysis, quals_exp_to_display = bs.initiate(text = jd_desc)

    # TF-IDF (important words)
    important_words_df = tfidf_df.head(15)
    important_words_plot = plot_df(y = 'feature_names', width = 'scores', df = important_words_df, plot_title = 'Important Words', xl = 'Score', yl = 'Keyword', figsz = (8,6))
    
    # Word Freq Plot
    word_freq_scores_plot =  plot_df(y = 'keyword', width = 'frequency', df = word_freq_scores, plot_title = 'Keyword Frequency', xl = 'Frequency', yl = 'Keyword', figsz = (8,6))
    
    return render_template(
            'analysis.html',
            job_desc = jd_desc, 
            important_words_df=important_words_df, 
            important_words_plot = important_words_plot, 
            word_freq_scores = word_freq_scores, 
            word_freq_scores_plot = word_freq_scores_plot, 
            entity_df = entity_df, 
            topic_df = topic_df, 
            skills =skills, 
            domain = domain_based_on_skill_analysis, 
            quals_exp= ",".join(quals_exp_to_display)
        )

@app.route('/download_page_as_pdf/<jd_desc>', methods=['GET'])
def download_page_as_pdf(jd_desc):
    tfidf_df, word_freq_scores, entity_df, topic_df, skills, domain_based_on_skill_analysis, quals_exp_to_display = bs.initiate(text = jd_desc)

    # TF-IDF (important words)
    important_words_df = tfidf_df.head(15)
    important_words_plot = plot_df(y = 'feature_names', width = 'scores', df = important_words_df, plot_title = 'Important Words', xl = 'Score', yl = 'Keyword', figsz = (8,6))
    
    # Word Freq Plot
    word_freq_scores_plot =  plot_df(y = 'keyword', width = 'frequency', df = word_freq_scores, plot_title = 'Keyword Frequency', xl = 'Frequency', yl = 'Keyword', figsz = (8,6))
    
    pdf_content = generate_pdf(
            {
                'job_desc': jd_desc,
                'important_words_df': important_words_df,
                'important_words_plot': important_words_plot,
                'word_freq_scores': word_freq_scores,
                'word_freq_scores_plot': word_freq_scores_plot,
                'entity_df': entity_df,
                'topic_df': topic_df,
                'skills': skills,
                'domain': domain_based_on_skill_analysis,
                'quals_exp': ", ".join(quals_exp_to_display),
            },
            'analysis.html'
        )

    response = make_response(pdf_content)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=downloaded_page.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True, port=3000)
