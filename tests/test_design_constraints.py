import unittest
import numpy as np
from design import constraint_mask, generate_design, constraint_description


class ExtendedConstraintTests(unittest.TestCase):
    schema=[dict(name='X',type='numeric'),dict(name='Z',type='numeric'),dict(name='M',type='categorical',categories=['a','b','c'])]
    points=np.array([[0,3,'a'],[1,2,'b'],[2,1,'c'],[3,0,'a']],dtype=object)

    def mask(self,rule):
        return constraint_mask(self.points,[rule],self.schema).tolist()

    def test_interval_inclusion_and_exclusion_at_boundaries(self):
        rule=dict(type='interval',left='X',operator='inside',lower=1,upper=2)
        self.assertEqual(self.mask(rule),[False,True,True,False])
        rule['operator']='outside'
        self.assertEqual(self.mask(rule),[True,False,False,True])

    def test_allowed_categories_and_numeric_membership(self):
        rule=dict(type='membership',left='M',operator='in',values=['a','c'])
        self.assertEqual(self.mask(rule),[True,False,True,True])
        rule['operator']='not_in'
        self.assertEqual(self.mask(rule),[False,True,False,False])
        self.assertEqual(self.mask(dict(type='membership',left='X',operator='in',values=[0,2])),[True,False,True,False])

    def test_weighted_sum_and_approximate_equality(self):
        terms=[dict(feature='X',coefficient=2),dict(feature='Z',coefficient=-1)]
        self.assertEqual(self.mask(dict(type='linear',terms=terms,operator='<=',value=0)),[True,True,False,False])
        self.assertEqual(self.mask(dict(type='linear',terms=terms,operator='==',value=.05,tolerance=.1)),[False,True,False,False])

    def test_absolute_gap_and_backward_compatible_comparison(self):
        rule=dict(type='distance',left='X',operator='>=',right=dict(kind='feature',value='Z'),value=2)
        self.assertEqual(self.mask(rule),[True,False,False,True])
        rule['operator']='<='
        self.assertEqual(self.mask(rule),[False,True,True,False])
        rule=dict(left='M',operator='!=',right=dict(kind='constant',value='a'))
        self.assertEqual(self.mask(rule),[False,True,True,False])

    def test_combined_constraints_apply_to_generated_design(self):
        rules=[dict(type='interval',left='X',operator='outside',lower=4,upper=6),
               dict(type='membership',left='M',operator='in',values=['a','c']),
               dict(type='linear',terms=[dict(feature='X',coefficient=1),dict(feature='Z',coefficient=2)],operator='<=',value=15),
               dict(type='distance',left='X',operator='>=',right=dict(kind='feature',value='Z'),value=2)]
        options=dict(count=40,seed=34,domain=[dict(name='X',minimum=0,maximum=10),dict(name='Z',minimum=0,maximum=10),dict(name='M',categories=['a','b','c'])],constraints=rules)
        result=generate_design(options,self.schema)
        self.assertEqual(len(result['rows']),40)
        points=np.array([r['values'] for r in result['rows']],dtype=object)
        self.assertTrue(constraint_mask(points,rules,self.schema).all())
        self.assertTrue(all(constraint_description(rule) for rule in rules))

    def test_invalid_constraint_parameters_are_rejected_without_sampling(self):
        rules=[dict(type='interval',left='M',operator='inside',lower=0,upper=1),
               dict(type='interval',left='X',operator='inside',lower=2,upper=1),
               dict(type='membership',left='M',operator='in',values=[]),
               dict(type='linear',terms=[dict(feature='M',coefficient=1)],operator='<=',value=2),
               dict(type='linear',terms=[dict(feature='X',coefficient=0)],operator='<=',value=2),
               dict(type='linear',terms=[dict(feature='X',coefficient=1)],operator='==',value=2,tolerance=-1),
               dict(type='distance',left='X',operator='>=',right=dict(kind='feature',value='Z'),value=-1),
               dict(type='__import__("os")',left='X')]
        for rule in rules:
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                constraint_mask(np.empty((0,3),dtype=object),[rule],self.schema)
