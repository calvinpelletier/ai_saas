from google.cloud import recaptchaenterprise_v1
from google.cloud.recaptchaenterprise_v1 import Assessment


def assess_captcha(project_id, recaptcha_key, token, recaptcha_action):
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()

    event = recaptchaenterprise_v1.Event()
    event.site_key = recaptcha_key
    event.token = token

    assessment = recaptchaenterprise_v1.Assessment()
    assessment.event = event

    project_name = f'projects/{project_id}'

    request = recaptchaenterprise_v1.CreateAssessmentRequest()
    request.assessment = assessment
    request.parent = project_name

    response = client.create_assessment(request)

    if not response.token_properties.valid:
        return None

    if response.token_properties.action != recaptcha_action:
        return None

    # https://cloud.google.com/recaptcha-enterprise/docs/interpret-assessment
    print('reasons:')
    for reason in response.risk_analysis.reasons:
        print(reason)
    print(f'score: {str(response.risk_analysis.score)}')

    return response.risk_analysis.score
