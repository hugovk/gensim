"""

Tests to check DTM math functions and Topic-Word, Doc-Topic proportions.

"""
import unittest
import logging

import numpy as np  # for arrays, array broadcasting etc.
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary
from gensim.test.utils import datapath


class TestLdaSeq(unittest.TestCase):
    # we are setting up a DTM model and fitting it, and checking topic-word and doc-topic results.
    def setUp(self):
        texts = [
            ['senior', 'studios', 'studios', 'studios', 'creators', 'award', 'mobile', 'currently',
             'challenges', 'senior', 'summary', 'senior', 'motivated', 'creative', 'senior'],
            ['performs', 'engineering', 'tasks', 'infrastructure', 'focusing', 'primarily',
             'programming', 'interaction', 'designers', 'engineers', 'leadership', 'teams',
             'teams', 'crews', 'responsibilities', 'engineering', 'quality', 'functional',
             'functional', 'teams', 'organizing', 'prioritizing', 'technical', 'decisions',
             'engineering', 'participates', 'participates', 'reviews', 'participates',
             'hiring', 'conducting', 'interviews'],
            ['feedback', 'departments', 'define', 'focusing', 'engineering', 'teams', 'crews',
             'facilitate', 'engineering', 'departments', 'deadlines', 'milestones', 'typically',
             'spends', 'designing', 'developing', 'updating', 'bugs', 'mentoring', 'engineers',
             'define', 'schedules', 'milestones', 'participating'],
            ['reviews', 'interviews', 'sized', 'teams', 'interacts', 'disciplines', 'knowledge',
             'skills', 'knowledge', 'knowledge', 'xcode', 'scripting', 'debugging', 'skills',
             'skills', 'knowledge', 'disciplines', 'animation', 'networking', 'expertise',
             'competencies', 'oral', 'skills', 'management', 'skills', 'proven', 'effectively',
             'teams', 'deadline', 'environment', 'bachelor', 'minimum', 'shipped', 'leadership',
             'teams', 'location', 'resumes', 'jobs', 'candidates', 'openings', 'jobs'],
            ['maryland', 'client', 'producers', 'electricity', 'operates', 'storage', 'utility',
             'retail', 'customers', 'engineering', 'consultant', 'maryland', 'summary', 'technical',
             'technology', 'departments', 'expertise', 'maximizing', 'output', 'reduces', 'operating',
             'participates', 'areas', 'engineering', 'conducts', 'testing', 'solve', 'supports',
             'environmental', 'understands', 'objectives', 'operates', 'responsibilities', 'handles',
             'complex', 'engineering', 'aspects', 'monitors', 'quality', 'proficiency', 'optimization',
             'recommendations', 'supports', 'personnel', 'troubleshooting', 'commissioning', 'startup',
             'shutdown', 'supports', 'procedure', 'operating', 'units', 'develops', 'simulations',
             'troubleshooting', 'tests', 'enhancing', 'solving', 'develops', 'estimates', 'schedules',
             'scopes', 'understands', 'technical', 'management', 'utilize', 'routine', 'conducts',
             'hazards', 'utilizing', 'hazard', 'operability', 'methodologies', 'participates', 'startup',
             'reviews', 'pssr', 'participate', 'teams', 'participate', 'regulatory', 'audits', 'define',
             'scopes', 'budgets', 'schedules', 'technical', 'management', 'environmental', 'awareness',
             'interfacing', 'personnel', 'interacts', 'regulatory', 'departments', 'input', 'objectives',
             'identifying', 'introducing', 'concepts', 'solutions', 'peers', 'customers', 'coworkers',
             'knowledge', 'skills', 'engineering', 'quality', 'engineering'],
            ['commissioning', 'startup', 'knowledge', 'simulators', 'technologies', 'knowledge',
             'engineering', 'techniques', 'disciplines', 'leadership', 'skills', 'proven',
             'engineers', 'oral', 'skills', 'technical', 'skills', 'analytically', 'solve',
             'complex', 'interpret', 'proficiency', 'simulation', 'knowledge', 'applications',
             'manipulate', 'applications', 'engineering'],
            ['calculations', 'programs', 'matlab', 'excel', 'independently', 'environment',
             'proven', 'skills', 'effectively', 'multiple', 'tasks', 'planning', 'organizational',
             'management', 'skills', 'rigzone', 'jobs', 'developer', 'exceptional', 'strategies',
             'junction', 'exceptional', 'strategies', 'solutions', 'solutions', 'biggest',
             'insurers', 'operates', 'investment'],
            ['vegas', 'tasks', 'electrical', 'contracting', 'expertise', 'virtually', 'electrical',
             'developments', 'institutional', 'utilities', 'technical', 'experts', 'relationships',
             'credibility', 'contractors', 'utility', 'customers', 'customer', 'relationships',
             'consistently', 'innovations', 'profile', 'construct', 'envision', 'dynamic', 'complex',
             'electrical', 'management', 'grad', 'internship', 'electrical', 'engineering',
             'infrastructures', 'engineers', 'documented', 'management', 'engineering',
             'quality', 'engineering', 'electrical', 'engineers', 'complex', 'distribution',
             'grounding', 'estimation', 'testing', 'procedures', 'voltage', 'engineering'],
            ['troubleshooting', 'installation', 'documentation', 'bsee', 'certification',
             'electrical', 'voltage', 'cabling', 'electrical', 'engineering', 'candidates',
             'electrical', 'internships', 'oral', 'skills', 'organizational', 'prioritization',
             'skills', 'skills', 'excel', 'cadd', 'calculation', 'autocad', 'mathcad',
             'skills', 'skills', 'customer', 'relationships', 'solving', 'ethic', 'motivation',
             'tasks', 'budget', 'affirmative', 'diversity', 'workforce', 'gender', 'orientation',
             'disability', 'disabled', 'veteran', 'vietnam', 'veteran', 'qualifying', 'veteran',
             'diverse', 'candidates', 'respond', 'developing', 'workplace', 'reflects', 'diversity',
             'communities', 'reviews', 'electrical', 'contracting', 'southwest', 'electrical', 'contractors'],
            ['intern', 'electrical', 'engineering', 'idexx', 'laboratories', 'validating', 'idexx',
             'integrated', 'hardware', 'entails', 'planning', 'debug', 'validation', 'engineers',
             'validation', 'methodologies', 'healthcare', 'platforms', 'brightest', 'solve',
             'challenges', 'innovation', 'technology', 'idexx', 'intern', 'idexx', 'interns',
             'supplement', 'interns', 'teams', 'roles', 'competitive', 'interns', 'idexx',
             'interns', 'participate', 'internships', 'mentors', 'seminars', 'topics', 'leadership',
             'workshops', 'relevant', 'planning', 'topics', 'intern', 'presentations', 'mixers',
             'applicants', 'ineligible', 'laboratory', 'compliant', 'idexx', 'laboratories', 'healthcare',
             'innovation', 'practicing', 'veterinarians', 'diagnostic', 'technology', 'idexx', 'enhance',
             'veterinarians', 'efficiency', 'economically', 'idexx', 'worldwide', 'diagnostic', 'tests',
             'tests', 'quality', 'headquartered', 'idexx', 'laboratories', 'employs', 'customers',
             'qualifications', 'applicants', 'idexx', 'interns', 'potential', 'demonstrated', 'portfolio',
             'recommendation', 'resumes', 'marketing', 'location', 'americas', 'verification', 'validation',
             'schedule', 'overtime', 'idexx', 'laboratories', 'reviews', 'idexx', 'laboratories',
             'nasdaq', 'healthcare', 'innovation', 'practicing', 'veterinarians'],
            ['location', 'duration', 'temp', 'verification', 'validation', 'tester', 'verification',
             'validation', 'middleware', 'specifically', 'testing', 'applications', 'clinical',
             'laboratory', 'regulated', 'environment', 'responsibilities', 'complex', 'hardware',
             'testing', 'clinical', 'analyzers', 'laboratory', 'graphical', 'interfaces', 'complex',
             'sample', 'sequencing', 'protocols', 'developers', 'correction', 'tracking',
             'tool', 'timely', 'troubleshoot', 'testing', 'functional', 'manual',
             'automated', 'participate', 'ongoing'],
            ['testing', 'coverage', 'planning', 'documentation', 'testing', 'validation',
             'corrections', 'monitor', 'implementation', 'recurrence', 'operating', 'statistical',
             'quality', 'testing', 'global', 'multi', 'teams', 'travel', 'skills', 'concepts',
             'waterfall', 'agile', 'methodologies', 'debugging', 'skills', 'complex', 'automated',
             'instrumentation', 'environment', 'hardware', 'mechanical', 'components', 'tracking',
             'lifecycle', 'management', 'quality', 'organize', 'define', 'priorities', 'organize',
             'supervision', 'aggressive', 'deadlines', 'ambiguity', 'analyze', 'complex', 'situations',
             'concepts', 'technologies', 'verbal', 'skills', 'effectively', 'technical', 'clinical',
             'diverse', 'strategy', 'clinical', 'chemistry', 'analyzer', 'laboratory', 'middleware',
             'basic', 'automated', 'testing', 'biomedical', 'engineering', 'technologists',
             'laboratory', 'technology', 'availability', 'click', 'attach'],
            ['scientist', 'linux', 'asrc', 'scientist', 'linux', 'asrc', 'technology',
             'solutions', 'subsidiary', 'asrc', 'engineering', 'technology', 'contracts'],
            ['multiple', 'agencies', 'scientists', 'engineers', 'management', 'personnel',
             'allows', 'solutions', 'complex', 'aeronautics', 'aviation', 'management', 'aviation',
             'engineering', 'hughes', 'technical', 'technical', 'aviation', 'evaluation',
             'engineering', 'management', 'technical', 'terminal', 'surveillance', 'programs',
             'currently', 'scientist', 'travel', 'responsibilities', 'develops', 'technology',
             'modifies', 'technical', 'complex', 'reviews', 'draft', 'conformity', 'completeness',
             'testing', 'interface', 'hardware', 'regression', 'impact', 'reliability',
             'maintainability', 'factors', 'standardization', 'skills', 'travel', 'programming',
             'linux', 'environment', 'cisco', 'knowledge', 'terminal', 'environment', 'clearance',
             'clearance', 'input', 'output', 'digital', 'automatic', 'terminal', 'management',
             'controller', 'termination', 'testing', 'evaluating', 'policies', 'procedure', 'interface',
             'installation', 'verification', 'certification', 'core', 'avionic', 'programs', 'knowledge',
             'procedural', 'testing', 'interfacing', 'hardware', 'regression', 'impact',
             'reliability', 'maintainability', 'factors', 'standardization', 'missions', 'asrc', 'subsidiaries',
             'affirmative', 'employers', 'applicants', 'disability', 'veteran', 'technology', 'location',
             'airport', 'bachelor', 'schedule', 'travel', 'contributor', 'management', 'asrc', 'reviews'],
            ['technical', 'solarcity', 'niche', 'vegas', 'overview', 'resolving', 'customer',
             'clients', 'expanding', 'engineers', 'developers', 'responsibilities', 'knowledge',
             'planning', 'adapt', 'dynamic', 'environment', 'inventive', 'creative', 'solarcity',
             'lifecycle', 'responsibilities', 'technical', 'analyzing', 'diagnosing', 'troubleshooting',
             'customers', 'ticketing', 'console', 'escalate', 'knowledge', 'engineering', 'timely',
             'basic', 'phone', 'functionality', 'customer', 'tracking', 'knowledgebase', 'rotation',
             'configure', 'deployment', 'sccm', 'technical', 'deployment', 'deploy', 'hardware',
             'solarcity', 'bachelor', 'knowledge', 'dell', 'laptops', 'analytical', 'troubleshooting',
             'solving', 'skills', 'knowledge', 'databases', 'preferably', 'server', 'preferably',
             'monitoring', 'suites', 'documentation', 'procedures', 'knowledge', 'entries', 'verbal',
             'skills', 'customer', 'skills', 'competitive', 'solar', 'package', 'insurance', 'vacation',
             'savings', 'referral', 'eligibility', 'equity', 'performers', 'solarcity', 'affirmative',
             'diversity', 'workplace', 'applicants', 'orientation', 'disability', 'veteran', 'careerrookie'],
            ['embedded', 'exelis', 'junction', 'exelis', 'embedded', 'acquisition', 'networking',
             'capabilities', 'classified', 'customer', 'motivated', 'develops', 'tests',
             'innovative', 'solutions', 'minimal', 'supervision', 'paced', 'environment', 'enjoys',
             'assignments', 'interact', 'multi', 'disciplined', 'challenging', 'focused', 'embedded',
             'developments', 'spanning', 'engineering', 'lifecycle', 'specification', 'enhancement',
             'applications', 'embedded', 'freescale', 'applications', 'android', 'platforms',
             'interface', 'customers', 'developers', 'refine', 'specifications', 'architectures'],
            ['java', 'programming', 'scripts', 'python', 'debug', 'debugging', 'emulators',
             'regression', 'revisions', 'specialized', 'setups', 'capabilities', 'subversion',
             'technical', 'documentation', 'multiple', 'engineering', 'techexpousa', 'reviews'],
            ['modeler', 'semantic', 'modeling', 'models', 'skills', 'ontology', 'resource',
             'framework', 'schema', 'technologies', 'hadoop', 'warehouse', 'oracle', 'relational',
             'artifacts', 'models', 'dictionaries', 'models', 'interface', 'specifications',
             'documentation', 'harmonization', 'mappings', 'aligned', 'coordinate', 'technical',
             'peer', 'reviews', 'stakeholder', 'communities', 'impact', 'domains', 'relationships',
             'interdependencies', 'models', 'define', 'analyze', 'legacy', 'models', 'corporate',
             'databases', 'architectural', 'alignment', 'customer', 'expertise', 'harmonization',
             'modeling', 'modeling', 'consulting', 'stakeholders', 'quality', 'models', 'storage',
             'agile', 'specifically', 'focus', 'modeling', 'qualifications', 'bachelors', 'accredited',
             'modeler', 'encompass', 'evaluation', 'skills', 'knowledge', 'modeling', 'techniques',
             'resource', 'framework', 'schema', 'technologies', 'unified', 'modeling', 'technologies',
             'schemas', 'ontologies', 'sybase', 'knowledge', 'skills', 'interpersonal', 'skills',
             'customers', 'clearance', 'applicants', 'eligibility', 'classified', 'clearance',
             'polygraph', 'techexpousa', 'solutions', 'partnership', 'solutions', 'integration'],
            ['technologies', 'junction', 'develops', 'maintains', 'enhances', 'complex', 'diverse',
             'intensive', 'analytics', 'algorithm', 'manipulation', 'management', 'documented',
             'individually', 'reviews', 'tests', 'components', 'adherence', 'resolves', 'utilizes',
             'methodologies', 'environment', 'input', 'components', 'hardware', 'offs', 'reuse', 'cots',
             'gots', 'synthesis', 'components', 'tasks', 'individually', 'analyzes', 'modifies',
             'debugs', 'corrects', 'integrates', 'operating', 'environments', 'develops', 'queries',
             'databases', 'repositories', 'recommendations', 'improving', 'documentation', 'develops',
             'implements', 'algorithms', 'functional', 'assists', 'developing', 'executing', 'procedures',
             'components', 'reviews', 'documentation', 'solutions', 'analyzing', 'conferring',
             'users', 'engineers', 'analyzing', 'investigating', 'areas', 'adapt', 'hardware',
             'mathematical', 'models', 'predict', 'outcome', 'implement', 'complex', 'database',
             'repository', 'interfaces', 'queries', 'bachelors', 'accredited', 'substituted',
             'bachelors', 'firewalls', 'ipsec', 'vpns', 'technology', 'administering', 'servers',
             'apache', 'jboss', 'tomcat', 'developing', 'interfaces', 'firefox', 'internet',
             'explorer', 'operating', 'mainframe', 'linux', 'solaris', 'virtual', 'scripting',
             'programming', 'oriented', 'programming', 'ajax', 'script', 'procedures', 'cobol',
             'cognos', 'fusion', 'focus', 'html', 'java', 'java', 'script', 'jquery', 'perl',
             'visual', 'basic', 'powershell', 'cots', 'cots', 'oracle', 'apex', 'integration',
             'competitive', 'package', 'bonus', 'corporate', 'equity', 'tuition', 'reimbursement',
             'referral', 'bonus', 'holidays', 'insurance', 'flexible', 'disability', 'insurance'],
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
        # initializing using own LDA sufficient statistics so that we get same results each time.
        sstats = np.loadtxt(datapath('DTM/sstats_test.txt'))
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        self.ldaseq = ldaseqmodel.LdaSeqModel(
            corpus=corpus, id2word=dictionary, num_topics=2,
            time_slice=[10, 10, 11], initialize='own', sstats=sstats
        )

    # testing topic word proportions
    def testTopicWord(self):

        topics = self.ldaseq.print_topics(0)
        expected_topic_word = [('skills', 0.035999999999999997)]
        self.assertEqual(topics[0][0][0], expected_topic_word[0][0])
        self.assertAlmostEqual(topics[0][0][1], expected_topic_word[0][1], places=2)

    # testing document-topic proportions
    def testDocTopic(self):
        doc_topic = self.ldaseq.doc_topics(0)
        expected_doc_topic = 0.00066577896138482028
        self.assertAlmostEqual(doc_topic[0], expected_doc_topic, places=2)

    def testDtypeBackwardCompatibility(self):
        ldaseq_3_0_1_fname = datapath('DTM/ldaseq_3_0_1_model')
        test_doc = [(547, 1), (549, 1), (552, 1), (555, 1)]
        expected_topics = [0.99751244, 0.00248756]

        # save model to use in test
        # self.ldaseq.save(ldaseq_3_0_1_fname)

        # load a model saved using a 3.0.1 version of Gensim
        model = ldaseqmodel.LdaSeqModel.load(ldaseq_3_0_1_fname)

        # and test it on a predefined document
        topics = model[test_doc]
        self.assertTrue(np.allclose(expected_topics, topics))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
