import os
import unittest
import numpy
import codecs
import pickle

from scipy import sparse
try:
    from sklearn.pipeline import Pipeline
    from sklearn import linear_model, cluster
    from sklearn.exceptions import NotFittedError
except ImportError:
    raise unittest.SkipTest("Test requires scikit-learn to be installed, which is not available")

from gensim.sklearn_api.ftmodel import FTTransformer
from gensim.sklearn_api.rpmodel import RpTransformer
from gensim.sklearn_api.ldamodel import LdaTransformer
from gensim.sklearn_api.lsimodel import LsiTransformer
from gensim.sklearn_api.ldaseqmodel import LdaSeqTransformer
from gensim.sklearn_api.w2vmodel import W2VTransformer
from gensim.sklearn_api.atmodel import AuthorTopicTransformer
from gensim.sklearn_api.d2vmodel import D2VTransformer
from gensim.sklearn_api.text2bow import Text2BowTransformer
from gensim.sklearn_api.tfidf import TfIdfTransformer
from gensim.sklearn_api.hdp import HdpTransformer
from gensim.sklearn_api.phrases import PhrasesTransformer
from gensim.corpora import mmcorpus, Dictionary
from gensim import matutils, models
from gensim.test.utils import datapath, common_texts

AZURE = bool(os.environ.get('PIPELINE_WORKSPACE'))

texts = [
    ['complier', 'system', 'computer'],
    ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'],
    ['graph', 'flow', 'network', 'graph'],
    ['loading', 'computer', 'system'],
    ['user', 'server', 'system'],
    ['tree', 'hamiltonian'],
    ['graph', 'trees'],
    ['computer', 'kernel', 'malfunction', 'computer'],
    ['server', 'system', 'computer'],
]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
author2doc = {
    'john': [0, 1, 2, 3, 4, 5, 6],
    'jane': [2, 3, 4, 5, 6, 7, 8],
    'jack': [0, 2, 4, 6, 8],
    'jill': [1, 3, 5, 7]
}

texts_new = texts[0:3]
author2doc_new = {
    'jill': [0],
    'bob': [0, 1],
    'sally': [1, 2]
}
dictionary_new = Dictionary(texts_new)
corpus_new = [dictionary_new.doc2bow(text) for text in texts_new]

texts_ldaseq = [
    [
        'senior', 'studios', 'studios', 'studios', 'creators', 'award', 'mobile', 'currently',
        'challenges', 'senior', 'summary', 'senior', 'motivated', 'creative', 'senior'
    ],
    [
        'performs', 'engineering', 'tasks', 'infrastructure', 'focusing', 'primarily', 'programming',
        'interaction', 'designers', 'engineers', 'leadership', 'teams', 'teams', 'crews', 'responsibilities',
        'engineering', 'quality', 'functional', 'functional', 'teams', 'organizing', 'prioritizing',
        'technical', 'decisions', 'engineering', 'participates', 'participates', 'reviews', 'participates',
        'hiring', 'conducting', 'interviews'
    ],
    [
        'feedback', 'departments', 'define', 'focusing', 'engineering', 'teams', 'crews', 'facilitate',
        'engineering', 'departments', 'deadlines', 'milestones', 'typically', 'spends', 'designing',
        'developing', 'updating', 'bugs', 'mentoring', 'engineers', 'define', 'schedules', 'milestones',
        'participating'
    ],
    [
        'reviews', 'interviews', 'sized', 'teams', 'interacts', 'disciplines', 'knowledge', 'skills',
        'knowledge', 'knowledge', 'xcode', 'scripting', 'debugging', 'skills', 'skills', 'knowledge',
        'disciplines', 'animation', 'networking', 'expertise', 'competencies', 'oral', 'skills',
        'management', 'skills', 'proven', 'effectively', 'teams', 'deadline', 'environment', 'bachelor',
        'minimum', 'shipped', 'leadership', 'teams', 'location', 'resumes', 'jobs', 'candidates',
        'openings', 'jobs'
    ],
    [
        'maryland', 'client', 'producers', 'electricity', 'operates', 'storage', 'utility', 'retail',
        'customers', 'engineering', 'consultant', 'maryland', 'summary', 'technical', 'technology',
        'departments', 'expertise', 'maximizing', 'output', 'reduces', 'operating', 'participates',
        'areas', 'engineering', 'conducts', 'testing', 'solve', 'supports', 'environmental', 'understands',
        'objectives', 'operates', 'responsibilities', 'handles', 'complex', 'engineering', 'aspects',
        'monitors', 'quality', 'proficiency', 'optimization', 'recommendations', 'supports', 'personnel',
        'troubleshooting', 'commissioning', 'startup', 'shutdown', 'supports', 'procedure', 'operating',
        'units', 'develops', 'simulations', 'troubleshooting', 'tests', 'enhancing', 'solving', 'develops',
        'estimates', 'schedules', 'scopes', 'understands', 'technical', 'management', 'utilize', 'routine',
        'conducts', 'hazards', 'utilizing', 'hazard', 'operability', 'methodologies', 'participates',
        'startup', 'reviews', 'pssr', 'participate', 'teams', 'participate', 'regulatory', 'audits',
        'define', 'scopes', 'budgets', 'schedules', 'technical', 'management', 'environmental', 'awareness',
        'interfacing', 'personnel', 'interacts', 'regulatory', 'departments', 'input', 'objectives',
        'identifying', 'introducing', 'concepts', 'solutions', 'peers', 'customers', 'coworkers', 'knowledge',
        'skills', 'engineering', 'quality', 'engineering'
    ],
    [
        'commissioning', 'startup', 'knowledge', 'simulators', 'technologies', 'knowledge', 'engineering',
        'techniques', 'disciplines', 'leadership', 'skills', 'proven', 'engineers', 'oral', 'skills',
        'technical', 'skills', 'analytically', 'solve', 'complex', 'interpret', 'proficiency', 'simulation',
        'knowledge', 'applications', 'manipulate', 'applications', 'engineering'
    ],
    [
        'calculations', 'programs', 'matlab', 'excel', 'independently', 'environment', 'proven', 'skills',
        'effectively', 'multiple', 'tasks', 'planning', 'organizational', 'management', 'skills', 'rigzone',
        'jobs', 'developer', 'exceptional', 'strategies', 'junction', 'exceptional', 'strategies', 'solutions',
        'solutions', 'biggest', 'insurers', 'operates', 'investment'
    ],
    [
        'vegas', 'tasks', 'electrical', 'contracting', 'expertise', 'virtually', 'electrical', 'developments',
        'institutional', 'utilities', 'technical', 'experts', 'relationships', 'credibility', 'contractors',
        'utility', 'customers', 'customer', 'relationships', 'consistently', 'innovations', 'profile',
        'construct', 'envision', 'dynamic', 'complex', 'electrical', 'management', 'grad', 'internship',
        'electrical', 'engineering', 'infrastructures', 'engineers', 'documented', 'management', 'engineering',
        'quality', 'engineering', 'electrical', 'engineers', 'complex', 'distribution', 'grounding',
        'estimation', 'testing', 'procedures', 'voltage', 'engineering'
    ],
    [
        'troubleshooting', 'installation', 'documentation', 'bsee', 'certification', 'electrical', 'voltage',
        'cabling', 'electrical', 'engineering', 'candidates', 'electrical', 'internships', 'oral', 'skills',
        'organizational', 'prioritization', 'skills', 'skills', 'excel', 'cadd', 'calculation', 'autocad',
        'mathcad', 'skills', 'skills', 'customer', 'relationships', 'solving', 'ethic', 'motivation', 'tasks',
        'budget', 'affirmative', 'diversity', 'workforce', 'gender', 'orientation', 'disability', 'disabled',
        'veteran', 'vietnam', 'veteran', 'qualifying', 'veteran', 'diverse', 'candidates', 'respond',
        'developing', 'workplace', 'reflects', 'diversity', 'communities', 'reviews', 'electrical',
        'contracting', 'southwest', 'electrical', 'contractors'
    ],
    [
        'intern', 'electrical', 'engineering', 'idexx', 'laboratories', 'validating', 'idexx', 'integrated',
        'hardware', 'entails', 'planning', 'debug', 'validation', 'engineers', 'validation', 'methodologies',
        'healthcare', 'platforms', 'brightest', 'solve', 'challenges', 'innovation', 'technology', 'idexx',
        'intern', 'idexx', 'interns', 'supplement', 'interns', 'teams', 'roles', 'competitive', 'interns',
        'idexx', 'interns', 'participate', 'internships', 'mentors', 'seminars', 'topics', 'leadership',
        'workshops', 'relevant', 'planning', 'topics', 'intern', 'presentations', 'mixers', 'applicants',
        'ineligible', 'laboratory', 'compliant', 'idexx', 'laboratories', 'healthcare', 'innovation',
        'practicing', 'veterinarians', 'diagnostic', 'technology', 'idexx', 'enhance', 'veterinarians',
        'efficiency', 'economically', 'idexx', 'worldwide', 'diagnostic', 'tests', 'tests', 'quality',
        'headquartered', 'idexx', 'laboratories', 'employs', 'customers', 'qualifications', 'applicants',
        'idexx', 'interns', 'potential', 'demonstrated', 'portfolio', 'recommendation', 'resumes', 'marketing',
        'location', 'americas', 'verification', 'validation', 'schedule', 'overtime', 'idexx', 'laboratories',
        'reviews', 'idexx', 'laboratories', 'nasdaq', 'healthcare', 'innovation', 'practicing', 'veterinarians'
    ],
    [
        'location', 'duration', 'temp', 'verification', 'validation', 'tester', 'verification', 'validation',
        'middleware', 'specifically', 'testing', 'applications', 'clinical', 'laboratory', 'regulated',
        'environment', 'responsibilities', 'complex', 'hardware', 'testing', 'clinical', 'analyzers',
        'laboratory', 'graphical', 'interfaces', 'complex', 'sample', 'sequencing', 'protocols', 'developers',
        'correction', 'tracking', 'tool', 'timely', 'troubleshoot', 'testing', 'functional', 'manual',
        'automated', 'participate', 'ongoing'
    ],
    [
        'testing', 'coverage', 'planning', 'documentation', 'testing', 'validation', 'corrections', 'monitor',
        'implementation', 'recurrence', 'operating', 'statistical', 'quality', 'testing', 'global', 'multi',
        'teams', 'travel', 'skills', 'concepts', 'waterfall', 'agile', 'methodologies', 'debugging', 'skills',
        'complex', 'automated', 'instrumentation', 'environment', 'hardware', 'mechanical', 'components',
        'tracking', 'lifecycle', 'management', 'quality', 'organize', 'define', 'priorities', 'organize',
        'supervision', 'aggressive', 'deadlines', 'ambiguity', 'analyze', 'complex', 'situations', 'concepts',
        'technologies', 'verbal', 'skills', 'effectively', 'technical', 'clinical', 'diverse', 'strategy',
        'clinical', 'chemistry', 'analyzer', 'laboratory', 'middleware', 'basic', 'automated', 'testing',
        'biomedical', 'engineering', 'technologists', 'laboratory', 'technology', 'availability', 'click',
        'attach'
    ],
    [
        'scientist', 'linux', 'asrc', 'scientist', 'linux', 'asrc', 'technology', 'solutions', 'subsidiary',
        'asrc', 'engineering', 'technology', 'contracts'
    ],
    [
        'multiple', 'agencies', 'scientists', 'engineers', 'management', 'personnel', 'allows', 'solutions',
        'complex', 'aeronautics', 'aviation', 'management', 'aviation', 'engineering', 'hughes', 'technical',
        'technical', 'aviation', 'evaluation', 'engineering', 'management', 'technical', 'terminal',
        'surveillance', 'programs', 'currently', 'scientist', 'travel', 'responsibilities', 'develops',
        'technology', 'modifies', 'technical', 'complex', 'reviews', 'draft', 'conformity', 'completeness',
        'testing', 'interface', 'hardware', 'regression', 'impact', 'reliability', 'maintainability',
        'factors', 'standardization', 'skills', 'travel', 'programming', 'linux', 'environment', 'cisco',
        'knowledge', 'terminal', 'environment', 'clearance', 'clearance', 'input', 'output', 'digital',
        'automatic', 'terminal', 'management', 'controller', 'termination', 'testing', 'evaluating', 'policies',
        'procedure', 'interface', 'installation', 'verification', 'certification', 'core', 'avionic',
        'programs', 'knowledge', 'procedural', 'testing', 'interfacing', 'hardware', 'regression', 'impact',
        'reliability', 'maintainability', 'factors', 'standardization', 'missions', 'asrc', 'subsidiaries',
        'affirmative', 'employers', 'applicants', 'disability', 'veteran', 'technology', 'location', 'airport',
        'bachelor', 'schedule', 'travel', 'contributor', 'management', 'asrc', 'reviews'
    ],
    [
        'technical', 'solarcity', 'niche', 'vegas', 'overview', 'resolving', 'customer', 'clients',
        'expanding', 'engineers', 'developers', 'responsibilities', 'knowledge', 'planning', 'adapt',
        'dynamic', 'environment', 'inventive', 'creative', 'solarcity', 'lifecycle', 'responsibilities',
        'technical', 'analyzing', 'diagnosing', 'troubleshooting', 'customers', 'ticketing', 'console',
        'escalate', 'knowledge', 'engineering', 'timely', 'basic', 'phone', 'functionality', 'customer',
        'tracking', 'knowledgebase', 'rotation', 'configure', 'deployment', 'sccm', 'technical', 'deployment',
        'deploy', 'hardware', 'solarcity', 'bachelor', 'knowledge', 'dell', 'laptops', 'analytical',
        'troubleshooting', 'solving', 'skills', 'knowledge', 'databases', 'preferably', 'server', 'preferably',
        'monitoring', 'suites', 'documentation', 'procedures', 'knowledge', 'entries', 'verbal', 'skills',
        'customer', 'skills', 'competitive', 'solar', 'package', 'insurance', 'vacation', 'savings',
        'referral', 'eligibility', 'equity', 'performers', 'solarcity', 'affirmative', 'diversity', 'workplace',
        'applicants', 'orientation', 'disability', 'veteran', 'careerrookie'
    ],
    [
        'embedded', 'exelis', 'junction', 'exelis', 'embedded', 'acquisition', 'networking', 'capabilities',
        'classified', 'customer', 'motivated', 'develops', 'tests', 'innovative', 'solutions', 'minimal',
        'supervision', 'paced', 'environment', 'enjoys', 'assignments', 'interact', 'multi', 'disciplined',
        'challenging', 'focused', 'embedded', 'developments', 'spanning', 'engineering', 'lifecycle',
        'specification', 'enhancement', 'applications', 'embedded', 'freescale', 'applications', 'android',
        'platforms', 'interface', 'customers', 'developers', 'refine', 'specifications', 'architectures'
    ],
    [
        'java', 'programming', 'scripts', 'python', 'debug', 'debugging', 'emulators', 'regression',
        'revisions', 'specialized', 'setups', 'capabilities', 'subversion', 'technical', 'documentation',
        'multiple', 'engineering', 'techexpousa', 'reviews'
    ],
    [
        'modeler', 'semantic', 'modeling', 'models', 'skills', 'ontology', 'resource', 'framework', 'schema',
        'technologies', 'hadoop', 'warehouse', 'oracle', 'relational', 'artifacts', 'models', 'dictionaries',
        'models', 'interface', 'specifications', 'documentation', 'harmonization', 'mappings', 'aligned',
        'coordinate', 'technical', 'peer', 'reviews', 'stakeholder', 'communities', 'impact', 'domains',
        'relationships', 'interdependencies', 'models', 'define', 'analyze', 'legacy', 'models', 'corporate',
        'databases', 'architectural', 'alignment', 'customer', 'expertise', 'harmonization', 'modeling',
        'modeling', 'consulting', 'stakeholders', 'quality', 'models', 'storage', 'agile', 'specifically',
        'focus', 'modeling', 'qualifications', 'bachelors', 'accredited', 'modeler', 'encompass', 'evaluation',
        'skills', 'knowledge', 'modeling', 'techniques', 'resource', 'framework', 'schema', 'technologies',
        'unified', 'modeling', 'technologies', 'schemas', 'ontologies', 'sybase', 'knowledge', 'skills',
        'interpersonal', 'skills', 'customers', 'clearance', 'applicants', 'eligibility', 'classified',
        'clearance', 'polygraph', 'techexpousa', 'solutions', 'partnership', 'solutions', 'integration'
    ],
    [
        'technologies', 'junction', 'develops', 'maintains', 'enhances', 'complex', 'diverse', 'intensive',
        'analytics', 'algorithm', 'manipulation', 'management', 'documented', 'individually', 'reviews',
        'tests', 'components', 'adherence', 'resolves', 'utilizes', 'methodologies', 'environment', 'input',
        'components', 'hardware', 'offs', 'reuse', 'cots', 'gots', 'synthesis', 'components', 'tasks',
        'individually', 'analyzes', 'modifies', 'debugs', 'corrects', 'integrates', 'operating',
        'environments', 'develops', 'queries', 'databases', 'repositories', 'recommendations', 'improving',
        'documentation', 'develops', 'implements', 'algorithms', 'functional', 'assists', 'developing',
        'executing', 'procedures', 'components', 'reviews', 'documentation', 'solutions', 'analyzing',
        'conferring', 'users', 'engineers', 'analyzing', 'investigating', 'areas', 'adapt', 'hardware',
        'mathematical', 'models', 'predict', 'outcome', 'implement', 'complex', 'database', 'repository',
        'interfaces', 'queries', 'bachelors', 'accredited', 'substituted', 'bachelors', 'firewalls',
        'ipsec', 'vpns', 'technology', 'administering', 'servers', 'apache', 'jboss', 'tomcat',
        'developing', 'interfaces', 'firefox', 'internet', 'explorer', 'operating', 'mainframe',
        'linux', 'solaris', 'virtual', 'scripting', 'programming', 'oriented', 'programming', 'ajax',
        'script', 'procedures', 'cobol', 'cognos', 'fusion', 'focus', 'html', 'java', 'java', 'script',
        'jquery', 'perl', 'visual', 'basic', 'powershell', 'cots', 'cots', 'oracle', 'apex', 'integration',
        'competitive', 'package', 'bonus', 'corporate', 'equity', 'tuition', 'reimbursement', 'referral',
        'bonus', 'holidays', 'insurance', 'flexible', 'disability', 'insurance'
    ],
    ['technologies', 'disability', 'accommodation', 'recruiter', 'techexpousa'],
    ['bank', 'river', 'shore', 'water'],
    ['river', 'water', 'flow', 'fast', 'tree'],
    ['bank', 'water', 'fall', 'flow'],
    ['bank', 'bank', 'water', 'rain', 'river'],
    ['river', 'water', 'mud', 'tree'],
    ['money', 'transaction', 'bank', 'finance'],
    ['bank', 'borrow', 'money'],
    ['bank', 'finance'],
    ['finance', 'money', 'sell', 'bank'],
    ['borrow', 'sell'],
    ['bank', 'loan', 'sell']
]
dictionary_ldaseq = Dictionary(texts_ldaseq)
corpus_ldaseq = [dictionary_ldaseq.doc2bow(text) for text in texts_ldaseq]

w2v_texts = [
    ['calculus', 'is', 'the', 'mathematical', 'study', 'of', 'continuous', 'change'],
    ['geometry', 'is', 'the', 'study', 'of', 'shape'],
    ['algebra', 'is', 'the', 'study', 'of', 'generalizations', 'of', 'arithmetic', 'operations'],
    ['differential', 'calculus', 'is', 'related', 'to', 'rates', 'of', 'change', 'and', 'slopes', 'of', 'curves'],
    ['integral', 'calculus', 'is', 'realted', 'to', 'accumulation', 'of', 'quantities', 'and',
     'the', 'areas', 'under', 'and', 'between', 'curves'],
    ['physics', 'is', 'the', 'natural', 'science', 'that', 'involves', 'the', 'study', 'of', 'matter',
     'and', 'its', 'motion', 'and', 'behavior', 'through', 'space', 'and', 'time'],
    ['the', 'main', 'goal', 'of', 'physics', 'is', 'to', 'understand', 'how', 'the', 'universe', 'behaves'],
    ['physics', 'also', 'makes', 'significant', 'contributions', 'through', 'advances', 'in', 'new',
     'technologies', 'that', 'arise', 'from', 'theoretical', 'breakthroughs'],
    ['advances', 'in', 'the', 'understanding', 'of', 'electromagnetism', 'or', 'nuclear', 'physics',
     'led', 'directly', 'to', 'the', 'development', 'of', 'new', 'products', 'that', 'have', 'dramatically',
     'transformed', 'modern', 'day', 'society']
]

d2v_sentences = [models.doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(w2v_texts)]

dict_texts = [' '.join(text) for text in common_texts]

phrases_sentences = common_texts + [
    ['graph', 'minors', 'survey', 'human', 'interface']
]

common_terms = ["of", "the", "was", "are"]
phrases_w_common_terms = [
    ['the', 'mayor', 'of', 'new', 'york', 'was', 'there'],
    ['the', 'mayor', 'of', 'new', 'orleans', 'was', 'there'],
    ['the', 'bank', 'of', 'america', 'offices', 'are', 'open'],
    ['the', 'bank', 'of', 'america', 'offices', 'are', 'closed']
]


class TestLdaWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        self.model = LdaTransformer(
            id2word=dictionary, num_topics=2, passes=100, minimum_probability=0, random_state=numpy.random.seed(0)
        )
        self.model.fit(corpus)

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
        doc = list(corpus)[0]  # transform only the first document
        transformed = self.model.transform(doc)
        expected = numpy.array([0.13, 0.87])
        passed = numpy.allclose(sorted(transformed[0]), sorted(expected), atol=1e-1)
        self.assertTrue(passed)

    def testConsistencyWithGensimModel(self):
        # training an LdaTransformer with `num_topics`=10
        self.model = LdaTransformer(
            id2word=dictionary, num_topics=10, passes=100, minimum_probability=0, random_state=numpy.random.seed(0)
        )
        self.model.fit(corpus)

        # training a Gensim LdaModel with the same params
        gensim_ldamodel = models.LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=10, passes=100,
            minimum_probability=0, random_state=numpy.random.seed(0)
        )

        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix_transformer_api = self.model.transform(bow)
        matrix_gensim_model = gensim_ldamodel[bow]
        # convert into dense representation to be able to compare with transformer output
        matrix_gensim_model_dense = matutils.sparse2full(matrix_gensim_model, 10)
        passed = numpy.allclose(matrix_transformer_api, matrix_gensim_model_dense, atol=1e-1)
        self.assertTrue(passed)

    def testCSRMatrixConversion(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        arr = numpy.array([[1, 2, 0], [0, 0, 3], [1, 0, 0]])
        sarr = sparse.csr_matrix(arr)
        newmodel = LdaTransformer(num_topics=2, passes=100)
        newmodel.fit(sarr)
        bow = [(0, 1), (1, 2), (2, 0)]
        transformed_vec = newmodel.transform(bow)
        expected_vec = numpy.array([0.12843782, 0.87156218])
        passed = numpy.allclose(transformed_vec, expected_vec, atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = LdaTransformer(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary([x.split() for x in data.data])
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline([('features', model,), ('classifier', clf)])
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
        self.assertGreaterEqual(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

        # updating multiple params
        param_dict = {"eval_every": 20, "decay": 0.7}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'eval_every'), 20)
        self.assertEqual(getattr(self.model.gensim_model, 'decay'), 0.7)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        texts_new = ['graph', 'eulerian']
        loaded_bow = model_load.id2word.doc2bow(texts_new)
        loaded_matrix = model_load.transform(loaded_bow)

        # sanity check for transformation operation
        self.assertEqual(loaded_matrix.shape[0], 1)
        self.assertEqual(loaded_matrix.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_bow = self.model.id2word.doc2bow(texts_new)
        original_matrix = self.model.transform(original_bow)
        passed = numpy.allclose(loaded_matrix, original_matrix, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        lda_wrapper = LdaTransformer(
            id2word=dictionary, num_topics=2, passes=100,
            minimum_probability=0, random_state=numpy.random.seed(0)
        )
        texts_new = ['graph', 'eulerian']
        bow = lda_wrapper.id2word.doc2bow(texts_new)
        self.assertRaises(NotFittedError, lda_wrapper.transform, bow)


class TestLsiWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        self.model = LsiTransformer(id2word=dictionary, num_topics=2)
        self.model.fit(corpus)

    def testTransform(self):
        texts_new = ['graph', 'eulerian']
        bow = self.model.id2word.doc2bow(texts_new)
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)
        texts_new = [['graph', 'eulerian'], ['server', 'flow'], ['path', 'system']]
        bow = []
        for i in texts_new:
            bow.append(self.model.id2word.doc2bow(i))
        matrix = self.model.transform(bow)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPartialFit(self):
        for i in range(10):
            self.model.partial_fit(X=corpus)  # fit against the model again
        doc = list(corpus)[0]  # transform only the first document
        transformed = self.model.transform(doc)
        expected = numpy.array([1.39, 0.0])
        passed = numpy.allclose(transformed[0], expected, atol=1)
        self.assertTrue(passed)

    def testPipeline(self):
        model = LsiTransformer(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary([x.split() for x in data.data])
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lsi = Pipeline([('features', model,), ('classifier', clf)])
        text_lsi.fit(corpus, data.target)
        score = text_lsi.score(corpus, data.target)
        self.assertGreater(score, 0.50)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

        # updating multiple params
        param_dict = {"chunksize": 10000, "decay": 0.9}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'chunksize'), 10000)
        self.assertEqual(getattr(self.model.gensim_model, 'decay'), 0.9)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        texts_new = ['graph', 'eulerian']
        loaded_bow = model_load.id2word.doc2bow(texts_new)
        loaded_matrix = model_load.transform(loaded_bow)

        # sanity check for transformation operation
        self.assertEqual(loaded_matrix.shape[0], 1)
        self.assertEqual(loaded_matrix.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_bow = self.model.id2word.doc2bow(texts_new)
        original_matrix = self.model.transform(original_bow)
        passed = numpy.allclose(loaded_matrix, original_matrix, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        lsi_wrapper = LsiTransformer(id2word=dictionary, num_topics=2)
        texts_new = ['graph', 'eulerian']
        bow = lsi_wrapper.id2word.doc2bow(texts_new)
        self.assertRaises(NotFittedError, lsi_wrapper.transform, bow)


class TestLdaSeqWrapper(unittest.TestCase):
    def setUp(self):
        self.model = LdaSeqTransformer(
            id2word=dictionary_ldaseq, num_topics=2, time_slice=[10, 10, 11], initialize='gensim'
        )
        self.model.fit(corpus_ldaseq)

    def testTransform(self):
        # transforming two documents
        docs = [list(corpus_ldaseq)[0], list(corpus_ldaseq)[1]]
        transformed_vecs = self.model.transform(docs)
        self.assertEqual(transformed_vecs.shape[0], 2)
        self.assertEqual(transformed_vecs.shape[1], self.model.num_topics)

        # transforming one document
        doc = list(corpus_ldaseq)[0]
        transformed_vecs = self.model.transform(doc)
        self.assertEqual(transformed_vecs.shape[0], 1)
        self.assertEqual(transformed_vecs.shape[1], self.model.num_topics)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        test_data = data.data[0:2]
        test_target = data.target[0:2]
        id2word = Dictionary([x.split() for x in test_data])
        corpus = [id2word.doc2bow(i.split()) for i in test_data]
        model = LdaSeqTransformer(id2word=id2word, num_topics=2, time_slice=[1, 1, 1], initialize='gensim')
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_ldaseq = Pipeline([('features', model,), ('classifier', clf)])
        text_ldaseq.fit(corpus, test_target)
        score = text_ldaseq.score(corpus, test_target)
        self.assertGreater(score, 0.50)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus_ldaseq)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = list(corpus_ldaseq)[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        ldaseq_wrapper = LdaSeqTransformer(num_topics=2)
        doc = list(corpus_ldaseq)[0]
        self.assertRaises(NotFittedError, ldaseq_wrapper.transform, doc)


class TestRpWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(13)
        self.model = RpTransformer(num_topics=2)
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.model.fit(self.corpus)

    def testTransform(self):
        # tranform two documents
        docs = [list(self.corpus)[0], list(self.corpus)[1]]
        matrix = self.model.transform(docs)
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

        # tranform one document
        doc = list(self.corpus)[0]
        matrix = self.model.transform(doc)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.num_topics)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        model = RpTransformer(num_topics=2)
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary([x.split() for x in data.data])
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_rp = Pipeline([('features', model,), ('classifier', clf)])
        text_rp.fit(corpus, data.target)
        score = text_rp.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(self.corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = list(self.corpus)[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.num_topics)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        rpmodel_wrapper = RpTransformer(num_topics=2)
        doc = list(self.corpus)[0]
        self.assertRaises(NotFittedError, rpmodel_wrapper.transform, doc)


class TestWord2VecWrapper(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = W2VTransformer(size=10, min_count=0, seed=42)
        self.model.fit(texts)

    def testTransform(self):
        # tranform multiple words
        words = []
        words = words + texts[0]
        matrix = self.model.transform(words)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.size)

        # tranform one word
        word = texts[0][0]
        matrix = self.model.transform(word)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.size)

    def testConsistencyWithGensimModel(self):
        # training a W2VTransformer
        self.model = W2VTransformer(size=10, min_count=0, seed=42)
        self.model.fit(texts)

        # training a Gensim Word2Vec model with the same params
        gensim_w2vmodel = models.Word2Vec(texts, size=10, min_count=0, seed=42)

        word = texts[0][0]
        vec_transformer_api = self.model.transform(word)  # vector returned by W2VTransformer
        vec_gensim_model = gensim_w2vmodel.wv[word]  # vector returned by Word2Vec
        passed = numpy.allclose(vec_transformer_api, vec_gensim_model, atol=1e-1)
        self.assertTrue(passed)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        model = W2VTransformer(size=10, min_count=1)
        model.fit(w2v_texts)

        class_dict = {'mathematics': 1, 'physics': 0}
        train_data = [
            ('calculus', 'mathematics'), ('mathematical', 'mathematics'),
            ('geometry', 'mathematics'), ('operations', 'mathematics'),
            ('curves', 'mathematics'), ('natural', 'physics'), ('nuclear', 'physics'),
            ('science', 'physics'), ('electromagnetism', 'physics'), ('natural', 'physics')
        ]
        train_input = [x[0] for x in train_data]
        train_target = [class_dict[x[1]] for x in train_data]

        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        clf.fit(model.transform(train_input), train_target)
        text_w2v = Pipeline([('features', model,), ('classifier', clf)])
        score = text_w2v.score(train_input, train_target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(negative=20)
        model_params = self.model.get_params()
        self.assertEqual(model_params["negative"], 20)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(texts)
        self.assertEqual(getattr(self.model.gensim_model, 'negative'), 20)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        word = texts[0][0]
        loaded_transformed_vecs = model_load.transform(word)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.size)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(word)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        w2vmodel_wrapper = W2VTransformer(size=10, min_count=0, seed=42)
        word = texts[0][0]
        self.assertRaises(NotFittedError, w2vmodel_wrapper.transform, word)


class TestAuthorTopicWrapper(unittest.TestCase):
    def setUp(self):
        self.model = AuthorTopicTransformer(id2word=dictionary, author2doc=author2doc, num_topics=2, passes=100)
        self.model.fit(corpus)

    def testTransform(self):
        # transforming multiple authors
        author_list = ['jill', 'jack']
        author_topics = self.model.transform(author_list)
        self.assertEqual(author_topics.shape[0], 2)
        self.assertEqual(author_topics.shape[1], self.model.num_topics)

        # transforming one author
        jill_topics = self.model.transform('jill')
        self.assertEqual(jill_topics.shape[0], 1)
        self.assertEqual(jill_topics.shape[1], self.model.num_topics)

    def testPartialFit(self):
        self.model.partial_fit(corpus_new, author2doc=author2doc_new)

        # Did we learn something about Sally?
        output_topics = self.model.transform('sally')
        sally_topics = output_topics[0]  # getting the topics corresponding to 'sally' (from the list of lists)
        self.assertTrue(all(sally_topics > 0))

    def testPipeline(self):
        # train the AuthorTopic model first
        model = AuthorTopicTransformer(id2word=dictionary, author2doc=author2doc, num_topics=10, passes=100)
        model.fit(corpus)

        # create and train clustering model
        clstr = cluster.MiniBatchKMeans(n_clusters=2)
        authors_full = ['john', 'jane', 'jack', 'jill']
        clstr.fit(model.transform(authors_full))

        # stack together the two models in a pipeline
        text_atm = Pipeline([('features', model,), ('cluster', clstr)])
        author_list = ['jane', 'jack', 'jill']
        ret_val = text_atm.predict(author_list)
        self.assertEqual(len(ret_val), len(author_list))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(num_topics=3)
        model_params = self.model.get_params()
        self.assertEqual(model_params["num_topics"], 3)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'num_topics'), 3)

        # updating multiple params
        param_dict = {"passes": 5, "iterations": 10}
        self.model.set_params(**param_dict)
        model_params = self.model.get_params()
        for key in param_dict.keys():
            self.assertEqual(model_params[key], param_dict[key])
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'passes'), 5)
        self.assertEqual(getattr(self.model.gensim_model, 'iterations'), 10)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        author_list = ['jill']
        loaded_author_topics = model_load.transform(author_list)

        # sanity check for transformation operation
        self.assertEqual(loaded_author_topics.shape[0], 1)
        self.assertEqual(loaded_author_topics.shape[1], self.model.num_topics)

        # comparing the original and loaded models
        original_author_topics = self.model.transform(author_list)
        passed = numpy.allclose(loaded_author_topics, original_author_topics, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        atmodel_wrapper = AuthorTopicTransformer(id2word=dictionary, author2doc=author2doc, num_topics=10, passes=100)
        author_list = ['jill', 'jack']
        self.assertRaises(NotFittedError, atmodel_wrapper.transform, author_list)


class TestD2VTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = D2VTransformer(min_count=1)
        self.model.fit(d2v_sentences)

    def testTransform(self):
        # tranform multiple documents
        docs = [w2v_texts[0], w2v_texts[1], w2v_texts[2]]
        matrix = self.model.transform(docs)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.size)

        # tranform one document
        doc = w2v_texts[0]
        matrix = self.model.transform(doc)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.size)

    def testFitTransform(self):
        model = D2VTransformer(min_count=1)

        # fit and transform multiple documents
        docs = [w2v_texts[0], w2v_texts[1], w2v_texts[2]]
        matrix = model.fit_transform(docs)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], model.size)

        # fit and transform one document
        doc = w2v_texts[0]
        matrix = model.fit_transform(doc)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], model.size)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(negative=20)
        model_params = self.model.get_params()
        self.assertEqual(model_params["negative"], 20)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(d2v_sentences)
        self.assertEqual(getattr(self.model.gensim_model, 'negative'), 20)

    def testPipeline(self):
        numpy.random.seed(0)  # set fixed seed to get similar values everytime
        model = D2VTransformer(min_count=1)
        model.fit(d2v_sentences)

        class_dict = {'mathematics': 1, 'physics': 0}
        train_data = [
            (['calculus', 'mathematical'], 'mathematics'), (['geometry', 'operations', 'curves'], 'mathematics'),
            (['natural', 'nuclear'], 'physics'), (['science', 'electromagnetism', 'natural'], 'physics')
        ]
        train_input = [x[0] for x in train_data]
        train_target = [class_dict[x[1]] for x in train_data]

        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        clf.fit(model.transform(train_input), train_target)
        text_w2v = Pipeline([('features', model,), ('classifier', clf)])
        score = text_w2v.score(train_input, train_target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = w2v_texts[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], 1)
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.size)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        passed = numpy.allclose(sorted(loaded_transformed_vecs), sorted(original_transformed_vecs), atol=1e-1)
        self.assertTrue(passed)

    def testConsistencyWithGensimModel(self):
        # training a D2VTransformer
        self.model = D2VTransformer(min_count=1)
        self.model.fit(d2v_sentences)

        # training a Gensim Doc2Vec model with the same params
        gensim_d2vmodel = models.Doc2Vec(d2v_sentences, min_count=1)

        doc = w2v_texts[0]
        vec_transformer_api = self.model.transform(doc)  # vector returned by D2VTransformer
        vec_gensim_model = gensim_d2vmodel[doc]  # vector returned by Doc2Vec
        passed = numpy.allclose(vec_transformer_api, vec_gensim_model, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        d2vmodel_wrapper = D2VTransformer(min_count=1)
        self.assertRaises(NotFittedError, d2vmodel_wrapper.transform, 1)


class TestText2BowTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = Text2BowTransformer()
        self.model.fit(dict_texts)

    def testTransform(self):
        # tranform one document
        doc = ['computer system interface time computer system']
        bow_vec = self.model.transform(doc)[0]
        expected_values = [1, 1, 2, 2]  # comparing only the word-counts
        values = [x[1] for x in bow_vec]
        self.assertEqual(sorted(expected_values), sorted(values))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(prune_at=1000000)
        model_params = self.model.get_params()
        self.assertEqual(model_params["prune_at"], 1000000)

    def testPipeline(self):
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        text2bow_model = Text2BowTransformer()
        lda_model = LdaTransformer(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline([('bow_model', text2bow_model), ('ldamodel', lda_model), ('classifier', clf)])
        text_lda.fit(data.data, data.target)
        score = text_lda.score(data.data, data.target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = dict_texts[0]
        loaded_transformed_vecs = model_load.transform(doc)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(doc)
        self.assertEqual(original_transformed_vecs, loaded_transformed_vecs)

    def testModelNotFitted(self):
        text2bow_wrapper = Text2BowTransformer()
        self.assertRaises(NotFittedError, text2bow_wrapper.transform, dict_texts[0])


class TestTfIdfTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = TfIdfTransformer(normalize=True)
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.model.fit(self.corpus)

    def testTransform(self):
        # tranform one document
        doc = corpus[0]
        transformed_doc = self.model.transform(doc)
        expected_doc = [[(0, 0.5773502691896257), (1, 0.5773502691896257), (2, 0.5773502691896257)]]
        self.assertTrue(numpy.allclose(transformed_doc, expected_doc))

        # tranform multiple documents
        docs = [corpus[0], corpus[1]]
        transformed_docs = self.model.transform(docs)
        expected_docs = [
            [(0, 0.5773502691896257), (1, 0.5773502691896257), (2, 0.5773502691896257)],
            [(3, 0.44424552527467476), (4, 0.44424552527467476), (5, 0.3244870206138555),
             (6, 0.44424552527467476), (7, 0.3244870206138555), (8, 0.44424552527467476)]
        ]
        self.assertTrue(numpy.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(numpy.allclose(transformed_docs[1], expected_docs[1]))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(smartirs='nnn')
        model_params = self.model.get_params()
        self.assertEqual(model_params["smartirs"], 'nnn')

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(self.corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'smartirs'), 'nnn')

    def testPipeline(self):
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary([x.split() for x in data.data])
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        tfidf_model = TfIdfTransformer()
        tfidf_model.fit(corpus)
        lda_model = LdaTransformer(num_topics=2, passes=10, minimum_probability=0, random_state=numpy.random.seed(0))
        numpy.random.mtrand.RandomState(1)  # set seed for getting same result
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_tfidf = Pipeline([('tfidf_model', tfidf_model), ('ldamodel', lda_model), ('classifier', clf)])
        text_tfidf.fit(corpus, data.target)
        score = text_tfidf.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = corpus[0]
        loaded_transformed_doc = model_load.transform(doc)

        # comparing the original and loaded models
        original_transformed_doc = self.model.transform(doc)
        self.assertEqual(original_transformed_doc, loaded_transformed_doc)

    def testModelNotFitted(self):
        tfidf_wrapper = TfIdfTransformer()
        self.assertRaises(NotFittedError, tfidf_wrapper.transform, corpus[0])


class TestHdpTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = HdpTransformer(id2word=dictionary, random_state=42)
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.model.fit(self.corpus)

    @unittest.skipIf(AZURE, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def testTransform(self):
        # tranform one document
        doc = self.corpus[0]
        transformed_doc = self.model.transform(doc)
        expected_doc = [
            [0.81043386270128193, 0.049357139518070477, 0.035840906753517532,
             0.026542006926698079, 0.019925705902962578, 0.014776690981729117, 0.011068909979528148]
        ]
        self.assertTrue(numpy.allclose(transformed_doc, expected_doc, atol=1e-2))

        # tranform multiple documents
        docs = [self.corpus[0], self.corpus[1]]
        transformed_docs = self.model.transform(docs)
        expected_docs = [
            [0.81043386270128193, 0.049357139518070477, 0.035840906753517532,
             0.026542006926698079, 0.019925705902962578, 0.014776690981729117, 0.011068909979528148],
            [0.03795908, 0.39542609, 0.50650585, 0.0151082, 0.01132749, 0., 0.]
        ]
        self.assertTrue(numpy.allclose(transformed_docs[0], expected_docs[0], atol=1e-2))
        self.assertTrue(numpy.allclose(transformed_docs[1], expected_docs[1], atol=1e-2))

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(var_converge=0.05)
        model_params = self.model.get_params()
        self.assertEqual(model_params["var_converge"], 0.05)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(self.corpus)
        self.assertEqual(getattr(self.model.gensim_model, 'm_var_converge'), 0.05)

    def testPipeline(self):
        with open(datapath('mini_newsgroup'), 'rb') as f:
            compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        data = cache
        id2word = Dictionary([x.split() for x in data.data])
        corpus = [id2word.doc2bow(i.split()) for i in data.data]
        model = HdpTransformer(id2word=id2word)
        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        text_lda = Pipeline([('features', model,), ('classifier', clf)])
        text_lda.fit(corpus, data.target)
        score = text_lda.score(corpus, data.target)
        self.assertGreater(score, 0.40)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = corpus[0]
        loaded_transformed_doc = model_load.transform(doc)

        # comparing the original and loaded models
        original_transformed_doc = self.model.transform(doc)
        self.assertTrue(numpy.allclose(original_transformed_doc, loaded_transformed_doc))

    def testModelNotFitted(self):
        hdp_wrapper = HdpTransformer(id2word=dictionary)
        self.assertRaises(NotFittedError, hdp_wrapper.transform, corpus[0])


class TestPhrasesTransformer(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.model = PhrasesTransformer(min_count=1, threshold=1)
        self.model.fit(phrases_sentences)

    def testTransform(self):
        # tranform one document
        doc = phrases_sentences[-1]
        phrase_tokens = self.model.transform(doc)[0]
        expected_phrase_tokens = ['graph_minors', 'survey', 'human_interface']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

    def testPartialFit(self):
        new_sentences = [
            ['world', 'peace', 'humans', 'world', 'peace', 'world', 'peace', 'people'],
            ['world', 'peace', 'people'],
            ['world', 'peace', 'humans']
        ]
        self.model.partial_fit(X=new_sentences)  # train model with new sentences

        doc = ['graph', 'minors', 'survey', 'human', 'interface', 'world', 'peace']
        phrase_tokens = self.model.transform(doc)[0]
        expected_phrase_tokens = ['graph_minors', 'survey', 'human_interface', 'world_peace']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(progress_per=5000)
        model_params = self.model.get_params()
        self.assertEqual(model_params["progress_per"], 5000)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(phrases_sentences)
        self.assertEqual(getattr(self.model.gensim_model, 'progress_per'), 5000)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = phrases_sentences[-1]
        loaded_phrase_tokens = model_load.transform(doc)

        # comparing the original and loaded models
        original_phrase_tokens = self.model.transform(doc)
        self.assertEqual(original_phrase_tokens, loaded_phrase_tokens)

    def testModelNotFitted(self):
        phrases_transformer = PhrasesTransformer()
        self.assertRaises(NotFittedError, phrases_transformer.transform, phrases_sentences[0])


class TestPhrasesTransformerCommonTerms(unittest.TestCase):
    def setUp(self):
        self.model = PhrasesTransformer(min_count=1, threshold=1, common_terms=common_terms)
        self.expected_transformations = [
            ['the', 'mayor_of_new', 'york', 'was', 'there'],
            ['the', 'mayor_of_new', 'orleans', 'was', 'there'],
            ['the', 'bank_of_america', 'offices', 'are', 'open'],
            ['the', 'bank_of_america', 'offices', 'are', 'closed']
        ]

    def testCompareToOld(self):
        with open(datapath("phrases-transformer-v3-5-0.pkl"), "rb") as old_phrases_transformer_pkl:
            old_phrases_transformer = pickle.load(old_phrases_transformer_pkl)
        doc = phrases_sentences[-1]
        phrase_tokens = old_phrases_transformer.transform(doc)[0]
        expected_phrase_tokens = ['graph_minors', 'survey', 'human_interface']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

        self.model.fit(phrases_sentences)
        new_phrase_tokens = self.model.transform(doc)[0]
        self.assertEqual(new_phrase_tokens, phrase_tokens)

    def testLoadNew(self):
        with open(datapath("phrases-transformer-new-v3-5-0.pkl"), "rb") as new_phrases_transformer_pkl:
            old_phrases_transformer = pickle.load(new_phrases_transformer_pkl)
        doc = phrases_sentences[-1]
        phrase_tokens = old_phrases_transformer.transform(doc)[0]
        expected_phrase_tokens = ['graph_minors', 'survey', 'human_interface']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

        self.model.fit(phrases_sentences)
        new_phrase_tokens = self.model.transform(doc)[0]
        self.assertEqual(new_phrase_tokens, phrase_tokens)

    def testFitAndTransform(self):
        self.model.fit(phrases_w_common_terms)

        transformed = self.model.transform(phrases_w_common_terms)
        self.assertEqual(transformed, self.expected_transformations)

    def testFitTransform(self):
        transformed = self.model.fit_transform(phrases_w_common_terms)
        self.assertEqual(transformed, self.expected_transformations)

    def testPartialFit(self):
        # fit half of the sentences
        self.model.fit(phrases_w_common_terms[:2])

        expected_transformations_0 = [
            ['the', 'mayor_of_new', 'york', 'was', 'there'],
            ['the', 'mayor_of_new', 'orleans', 'was', 'there'],
            ['the', 'bank', 'of', 'america', 'offices', 'are', 'open'],
            ['the', 'bank', 'of', 'america', 'offices', 'are', 'closed']
        ]
        # transform all sentences, second half should be same as original
        transformed_0 = self.model.transform(phrases_w_common_terms)
        self.assertEqual(transformed_0, expected_transformations_0)

        # fit remaining sentences, result should be the same as in the other tests
        self.model.partial_fit(phrases_w_common_terms[2:])
        transformed_1 = self.model.fit_transform(phrases_w_common_terms)
        self.assertEqual(transformed_1, self.expected_transformations)

        new_phrases = [['offices', 'are', 'open'], ['offices', 'are', 'closed']]
        self.model.partial_fit(new_phrases)
        expected_transformations_2 = [
            ['the', 'mayor_of_new', 'york', 'was', 'there'],
            ['the', 'mayor_of_new', 'orleans', 'was', 'there'],
            ['the', 'bank_of_america', 'offices_are_open'],
            ['the', 'bank_of_america', 'offices_are_closed']
        ]
        transformed_2 = self.model.transform(phrases_w_common_terms)
        self.assertEqual(transformed_2, expected_transformations_2)


# specifically test pluggable scoring in Phrases, because possible pickling issues with function parameter

# this is intentionally in main rather than a class method to support pickling
# all scores will be 1
def dumb_scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    return 1


class TestPhrasesTransformerCustomScorer(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)

        self.model = PhrasesTransformer(min_count=1, threshold=.9, scoring=dumb_scorer)
        self.model.fit(phrases_sentences)

    def testTransform(self):
        # tranform one document
        doc = phrases_sentences[-1]
        phrase_tokens = self.model.transform(doc)[0]
        expected_phrase_tokens = ['graph_minors', 'survey_human', 'interface']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

    def testPartialFit(self):
        new_sentences = [
            ['world', 'peace', 'humans', 'world', 'peace', 'world', 'peace', 'people'],
            ['world', 'peace', 'people'],
            ['world', 'peace', 'humans']
        ]
        self.model.partial_fit(X=new_sentences)  # train model with new sentences

        doc = ['graph', 'minors', 'survey', 'human', 'interface', 'world', 'peace']
        phrase_tokens = self.model.transform(doc)[0]
        expected_phrase_tokens = ['graph_minors', 'survey_human', 'interface', 'world_peace']
        self.assertEqual(phrase_tokens, expected_phrase_tokens)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(progress_per=5000)
        model_params = self.model.get_params()
        self.assertEqual(model_params["progress_per"], 5000)

        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(phrases_sentences)
        self.assertEqual(getattr(self.model.gensim_model, 'progress_per'), 5000)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        doc = phrases_sentences[-1]
        loaded_phrase_tokens = model_load.transform(doc)

        # comparing the original and loaded models
        original_phrase_tokens = self.model.transform(doc)
        self.assertEqual(original_phrase_tokens, loaded_phrase_tokens)

    def testModelNotFitted(self):
        phrases_transformer = PhrasesTransformer()
        self.assertRaises(NotFittedError, phrases_transformer.transform, phrases_sentences[0])


class TestFastTextWrapper(unittest.TestCase):
    def setUp(self):
        self.model = FTTransformer(size=10, min_count=0, seed=42, bucket=5000)
        self.model.fit(texts)

    def testTransform(self):
        # tranform multiple words
        words = []
        words = words + texts[0]
        matrix = self.model.transform(words)
        self.assertEqual(matrix.shape[0], 3)
        self.assertEqual(matrix.shape[1], self.model.size)

        # tranform one word
        word = texts[0][0]
        matrix = self.model.transform(word)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.shape[1], self.model.size)

        # verify oov-word vector retrieval
        invocab_vec = self.model.transform("computer")  # invocab word
        self.assertEqual(invocab_vec.shape[0], 1)
        self.assertEqual(invocab_vec.shape[1], self.model.size)

        oov_vec = self.model.transform('compute')  # oov word
        self.assertEqual(oov_vec.shape[0], 1)
        self.assertEqual(oov_vec.shape[1], self.model.size)

    def testConsistencyWithGensimModel(self):
        # training a FTTransformer
        self.model = FTTransformer(size=10, min_count=0, seed=42, workers=1, bucket=5000)
        self.model.fit(texts)

        # training a Gensim FastText model with the same params
        gensim_ftmodel = models.FastText(texts, size=10, min_count=0, seed=42, workers=1, bucket=5000)

        # vectors returned by FTTransformer
        vecs_transformer_api = self.model.transform(
                [text for text_list in texts for text in text_list])
        # vectors returned by FastText
        vecs_gensim_model = [gensim_ftmodel.wv[text] for text_list in texts for text in text_list]
        passed = numpy.allclose(vecs_transformer_api, vecs_gensim_model)
        self.assertTrue(passed)

        # test for out of vocab words
        oov_words = ["compute", "serve", "sys", "net"]
        vecs_transformer_api = self.model.transform(oov_words)  # vector returned by FTTransformer
        vecs_gensim_model = [gensim_ftmodel.wv[word] for word in oov_words]  # vector returned by FastText
        passed = numpy.allclose(vecs_transformer_api, vecs_gensim_model)
        self.assertTrue(passed)

    def testPipeline(self):
        model = FTTransformer(size=10, min_count=1, bucket=5000)
        model.fit(w2v_texts)

        class_dict = {'mathematics': 1, 'physics': 0}
        train_data = [
            ('calculus', 'mathematics'), ('mathematical', 'mathematics'),
            ('geometry', 'mathematics'), ('operations', 'mathematics'),
            ('curves', 'mathematics'), ('natural', 'physics'), ('nuclear', 'physics'),
            ('science', 'physics'), ('electromagnetism', 'physics'), ('natural', 'physics')
        ]
        train_input = [x[0] for x in train_data]
        train_target = [class_dict[x[1]] for x in train_data]

        clf = linear_model.LogisticRegression(penalty='l2', C=0.1)
        clf.fit(model.transform(train_input), train_target)
        text_ft = Pipeline([('features', model,), ('classifier', clf)])
        score = text_ft.score(train_input, train_target)
        self.assertGreater(score, 0.40)

    def testSetGetParams(self):
        # updating only one param
        self.model.set_params(negative=20)
        model_params = self.model.get_params()
        self.assertEqual(model_params["negative"], 20)
        # verify that the attributes values are also changed for `gensim_model` after fitting
        self.model.fit(texts)
        self.assertEqual(getattr(self.model.gensim_model, 'negative'), 20)

    def testPersistence(self):
        model_dump = pickle.dumps(self.model)
        model_load = pickle.loads(model_dump)

        # pass all words in one list
        words = [word for text_list in texts for word in text_list]
        loaded_transformed_vecs = model_load.transform(words)

        # sanity check for transformation operation
        self.assertEqual(loaded_transformed_vecs.shape[0], len(words))
        self.assertEqual(loaded_transformed_vecs.shape[1], model_load.size)

        # comparing the original and loaded models
        original_transformed_vecs = self.model.transform(words)
        passed = numpy.allclose(loaded_transformed_vecs, original_transformed_vecs, atol=1e-1)
        self.assertTrue(passed)

    def testModelNotFitted(self):
        ftmodel_wrapper = FTTransformer(size=10, min_count=0, seed=42, bucket=5000)
        word = texts[0][0]
        self.assertRaises(NotFittedError, ftmodel_wrapper.transform, word)


if __name__ == '__main__':
    unittest.main()
