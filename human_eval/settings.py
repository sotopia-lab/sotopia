from os import environ


SESSION_CONFIGS = [
    dict(
        name='sotopia_pilot_study',
        display_name='social interaction qualification test',
        app_sequence=['sotopia_pilot_study', 'pilot_study_payment_info'],
        num_demo_participants=1,
    ),
    dict(
        name='sotopia_official_study',
        display_name='social interaction official test',
        app_sequence=['sotopia_official_study', 'official_study_payment_info'],
        num_demo_participants=1,
    )
]



# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc=""
)

PARTICIPANT_FIELDS = ['expiry'] # to use the timer (haofei)
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

DEBUG = False # TODO(haofeiyu): to control whether the annotators could see the debug_info or not

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ROOMS = [
    dict(
        name='econ101',
        display_name='Econ 101 class',
        participant_label_file='_rooms/econ101.txt',
    ),
    dict(name='live_demo', display_name='Room for live demo (no participant labels)'),
]

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """
Here are some oTree games.
"""


SECRET_KEY = '4197606110806'

INSTALLED_APPS = ['otree']
