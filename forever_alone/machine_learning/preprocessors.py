import types

import numpy

import pandas
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, robust_scale, LabelBinarizer, Imputer, RobustScaler
from sklearn.svm import SVC


class FACleaner(TransformerMixin):
    data = {
        "race": {
            "mixed race": [True, "mix", "half", " and ", "multi"],
            "white non-hispanic": [False, "caucasian", "european", "white non-hispanic"],
            numpy.nan: [False, "helicopterkin",
                        "first two answers. gender is androgyne, not male; sexuality is asexual, not bi."],
            "middle eastern": [False, "pakistani"]
        },

        "job_title": {
            "unemployed": [False, "*", '-', '--', '---', '.', '...', '/', "disabled", "unemployed",
                           "i don't have a job", "no job", "n/a", 'na', 'nan', 'i have no job', 'neet', 'neet parasite',
                           'no', 'no job', 'none', 'none (?)', 'not employed', 'not working', 'nothing', 'out of work',
                           'professional neet', "funemployed", "currently unemployed"],
            "student": [True, "high school senior", "high school senior ", "masters", "intern", "bio major",
                        "cgma sudent", "student", "college", "freshman", "junior ba", "undergraduate",
                        " undergraduate."],
            numpy.nan: [False, "peon", "dunno", "erdet", "guy", "ggg", "grunt", "junkie", "j", "loser", "bum",
                        'not telling', 'not disclosing this', "rather not say", "rn", "s", "shit kicker", 'sugar baby',
                        'super pooper', 'u serious?', "t", 'what?', 'why is this obligatory?',
                        'your survey question design is bad and you should feel bad.', '\xf0\x9f\x8c\x9a', 'useless'],
            "ICT": [False, "software development", "data and development", "it engineer", "software eng.",
                    "application developer", "senior developer", "system administrator", "senior software developer",
                    "it technician", "data entry", "consultant it", "it operations", "game programmer", "online seller",
                    "system analyst associate", "principal software engineer", "web developpeur", "admin", "programmer",
                    "coder", "software developer", "software engineer", "systems analyst", "it", "i/t support",
                    "software full stack developer"],
            "technology": [False, "freelance technical theatre", "mechanical drafter", "trainee mechanical engineer",
                           "aviation electricians mate", "optometric tech", "architectural technician", "engineer",
                           "production engineer", "technician", "mechanical engineer"],
            "government": [False, "popo", "2nd lieutenant", "law (enforcement) officer",
                           "zivildiener (austrian stuff)"],
            "financial/sales": [False, "store clerk", "accounts receivable", "broker", "retail associate", "retail",
                                "financial analyst", "economist", "retail worker", "sales clerk", "call operator",
                                "call center agent", "account manager", "audit associate", "marketing",
                                "sales associate", "cashier", "insurance agent.", "accountant", "sales", "auditor",
                                "sales manager", "business representative (have in mind that i am from poland)"],
            "managing": [False, "president",
                         "operations manager (i.e. i do what my father tells me to do in our family company).",
                         "management", "program manager, non-profit", "third manager", "ceo", "manager", "supervisor",
                         "boss", "operations executive", "coordinator", "surveyor", "coo"],
            "academic": [False, "scientist", "research associate", "research", "researcher", "research assistant"],
            "education": [False, "teacher", "part time computer tutor", "tutor", "language teacher"],
            "art": [False, "artist", "musician", "freelance writer"],
            "self-employed": [False, "entrepreneur", "butcher", "storekeeper", "self employed"],
            "laborer": [False, "vehicle prep", "terminal worker", "freight associate", "press operator", "fabricator",
                        "mechanic", "home depot order fulfillment", "worker", "delivery guy", "freight broker",
                        "cleaner", "courier", "stockroom assistant", "stocker", "factory worker", "warehouse associate",
                        "janitor"],
            "administration": [False, "administrator", "administrative officer", "clerk", "admin assistant",
                               "administrative assistant"],
            "catering industry": [False, "waiter", "chef", "waitress", "cook", "kitchen staff"],
            "design": [False, "product designer", "web desihner", "designer", "graphic designer", ],
            "other": [False, "guest advisor", "consultant", "lawyer", "librarian", "languages", "customer service",
                      "receptionist", "service desk analyst", "vehicle", "prep", "medical services",
                      "intermediate call center representative", "therapy", "interpreter", "developer", "real estate",
                      "nurse", "user-carer", "farm hand", "iternent peagogue", "pa", "carer", "delivery manager",
                      "truck driver", "postman", "castor", "philosopher", "lawm maintenance", "transport",
                      "cst analyst", "part time paperboy", "assistant caretaker", "job", "production editor",
                      "lifeguard", "screener", "case worker", "herder", "flight attendant", "jpo;", "analyst", "usher",
                      "lackey"]
        },
        "what_help_from_others": {
            numpy.nan: ["i lost faith and hope", "more one night stands", "coping with being beyond help",
                        "it would be nice but why would a woman choose me?  there are plenty better", "kill me", "like",
                        'there is no way that they can help. they only give useless advice like "just be more confident".',
                        "i'm not fa lol", "im on my own", "someone to kill.me", "pity fuck", "invite me to shit please",
                        "more general stuff",
                        "i used to want all of those things. now it is too late for any of them to make a difference.", ],
            "friendship": ["to be seen & treated like a normal person", "emotional support", "friends", "trust",
                           "just more friends/greater social life in general tbh",
                           "just want mates that make it look like i fit in", "shoulder to cry on",
                           "someone to hang out with me where i can meet met (e.g. class",
                           "free event) and sell me up"],
            "i don't want help": ["i don't want any help", "i don't want any help. i can't even talk about it.",
                                  "i want help but i am not sure what kind. i always think it would be nice if a woman would approch me but thats not realistic."],
            "i don't know": ["i have no idea", "not sure what would help",
                             "i don't really know. something magical that would make me happy.", "any help", "anything",
                             "i don't even know.."],
            "therapy": ["maybe to learn how to be happy", "money for rehab or sober living home",
                        "help getting a better job", "teach me how to talk to people", "social skills training"]
        },
        "edu_level": {
            "bachelor's degree": [False, "bachelor\xe2\x80\x99s degree"],
            "master's degree": [False, "master\xe2\x80\x99s degree"]
        },
        "improve_yourself_how": {
            "dieting": ["diets", "eat healthy", "changed my hairstyle and lost weight", "eat more",
                        "lose weight", "losing weight through better diet. no exercise yet.", "losing weight",
                        "started losing weight"],
            "join clubs/social clubs/meet ups": ["join clubs/socual clubs/meet ups", "go to bars", "partying",
                                                 "clubbing",
                                                 "forced myself to free events or events w/acquintances advertised in fb; been part of a blog for five years and met some members in person"],
            "cosmetic surgery": ["cosmetic survey"],
            "hobbies": ["volunteering", "social activism", "traveling", "play live in a band.",
                        "joined a german language learning course.", "practice various skills", "learn new skills",
                        "develop hobbies", "such as writing and piano.", "hobbies - music", "languages", "calligraphy"],
            numpy.nan: ["just b confident", "etc", "trying to accept my fate.", "dealer",
                        "keeps you in touch with a lot of different people"],
            "none": ["None", "nope not fa"],
            "appearance": ["hair", "makeup", "clothing", "fashion makeup personality etc",
                           "change of wardrobe"],
            "other": ["other excercise", "non-physical forms of improvement", "called suicide hotlines",
                      "work out at home", "talking to people", "change my lifestyle", "self analysis",
                      "joined the us navy", "living in a student house (at dorms)"],
            "medication": ["started taking some medication", "diet pills"]
        }
    }



    @staticmethod
    def _list_replace(value, dct):
        if isinstance(value, types.StringTypes):
            orig_lst = value.lower().strip().split(", ")
        else:
            orig_lst = list(value)
        for new_value, lst in dct.items():
            for old_value in lst:
                try:
                    index = orig_lst.index(old_value)
                    orig_lst[index] = new_value
                except ValueError:
                    pass
        return tuple(orig_lst)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self,  ds):
        for column_name, dct in self.data.items():
            if column_name in ("improve_yourself_how", "what_help_from_others"):
                ds.loc[:,column_name] = ds[column_name].apply(lambda x: self._list_replace(x, dct))
            elif column_name in ("friends", "age"):
                ds.loc[:,column_name] = ds.loc[:,column_name].apply(lambda x: int(x) if x >= 0 else 0)
            else:
                ds.loc[:, column_name] = ds.loc[:, column_name].apply(lambda x: str(x).lower().strip())
                for new_value, lst in dct.items():
                    for index in range(1, len(lst)):
                        to_replace = lst[index]
                        regex = False
                        if lst[0]:
                            to_replace = ".*" + to_replace + ".*"
                            regex = True
                        ds.loc[:,column_name] = ds.loc[:,column_name].replace(to_replace=to_replace, value=new_value, regex=regex)
        return ds

class FAImputer(TransformerMixin):

    def _find_most_frequent_value_special(self, series):
        frequency = {}
        for lst, count in series.value_counts().iteritems():
            for name in lst:
                try:
                    frequency[name] += count
                except KeyError:
                    frequency[name] = count
        return max(zip(frequency.values(), frequency.keys()))

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, raw_ds):
        for column in raw_ds.columns:
            if column in ("what_help_from_others", "improve_yourself_how"):
                _, most_frequent_value = self._find_most_frequent_value_special(raw_ds.loc[:, column])
                raw_ds.loc[:, column] = raw_ds.loc[:, column].apply(
                    lambda x: tuple(
                        [most_frequent_value if not isinstance(y, types.StringTypes) and numpy.isnan(y) else y for y in
                         x])
                )
            else:
                raw_ds.loc[:, column] = raw_ds.loc[:, column].fillna(
                    raw_ds.loc[:, column].value_counts().index[0]
                )

            #assert raw_ds.loc[:, column].value_counts().sum() == 469
        return raw_ds

class FATransformer(TransformerMixin):

    dummy_columns = None
    encoders = {}
    to_be_dummied = ("gender", "sexuallity", "race", "employment", "pay_for_sex", "job_title")
    dummy_extra = ("job_title",)
    labels_with_order = {
        "bodyweight": ['Underweight', 'Normal weight', 'Overweight', 'Obese'],
        "income": ['$0', '$1 to $10,000', '$10,000 to $19,999', '$20,000 to $29,999', '$30,000 to $39,999',
                   '$40,000 to $49,999', '$50,000 to $74,999',
                   '$75,000 to $99,999', '$100,000 to $124,999', '$125,000 to $149,999', '$150,000 to $174,999',
                   '$174,999 to $199,999', '$200,000 or more'],
        "edu_level": ['some high school, no diploma',
                      'high school graduate, diploma or the equivalent (for example: ged)',
                      'some college, no degree', 'trade/technical/vocational training', 'associate degree',
                      'professional degree', "bachelor's degree", "master's degree", 'doctorate degree']
    }


    def __init__(self):
        self._fit_ordered_labels()


    def fit(self, X, y=None, **fit_params):
        self.dummy_columns = self._fit_dummies(X).columns
        self._fit_multi_columns(X)
        self._fit_binary(X)
        return self


    def _fit_dummies(self, X):
        new_X = pandas.get_dummies(X.loc[:, self.to_be_dummied])
        self.dummy_columns = new_X.columns
        return new_X

    def _transform_dummies(self, X, replace_with="other"):
        ser = X.loc[:,"job_title"].apply(lambda x: x if x in self.dummy_columns else replace_with)
        return pandas.get_dummies(ser)


    def _fit_ordered_labels(self):
        for column, values in self.labels_with_order.items():
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(values)

    def _fit_multi_columns(self, X):
        for column in ("what_help_from_others", "improve_yourself_how"):
            classes = self._get_multi_unique_values(X.loc[:, column])
            classes.add("other")
            self.encoders[column] = MultiLabelBinarizer(classes=tuple(classes))
            self.encoders[column].fit(X.loc[:,column])

    def _get_multi_unique_values(self, column_series):
        unique = set()
        for lst in column_series.unique():
            for value in lst:
                unique.add(value)
        return unique


    def _replace_multi_unique_values(self, X, column, values_to_replace, replace_with="other"):
        series = X.loc[:, column].copy()
        for index, tup in series.iteritems():
            lst = list(tup)
            lst = [replace_with if x in values_to_replace else x for x in lst]
            series.at[index] = tuple(lst)
        return series

    def _fit_binary(self, X):
        for column in ("virgin", "prostitution_legal", "social_fear", "depressed"):
            self.encoders[column] = LabelBinarizer()
            self.encoders[column].fit(X.loc[:, column])

    def transform(self, raw_ds):
        ds = pandas.DataFrame()
        ds = pandas.concat([ds, self._transform_dummies(raw_ds)], axis=1, join='outer')
        for column in ("bodyweight","income", "edu_level"):
            ds.loc[:,column] = self.encoders[column].transform(raw_ds.loc[:, column])

        for column in ("what_help_from_others", "improve_yourself_how"):
            new_columns = [x for x in self._get_multi_unique_values(raw_ds.loc[:, column]) if x not in self.encoders[column].classes_]
            ds[column] = self._replace_multi_unique_values(raw_ds, column, new_columns)
            multi = pandas.DataFrame(self.encoders[column].transform(ds.loc[:,column]), columns=self.encoders[column].classes_,
                                     index=raw_ds.index)

            multi.columns = multi.columns.map(lambda name: column + '_' + str(name))
            del ds[column]
            ds = pandas.concat([ds, multi], axis=1, join='outer')

        for column, values in self.labels_with_order.items():
            ds.loc[:, column] = self.encoders[column].transform(raw_ds.loc[:, column])

        for column in ("virgin", "prostitution_legal", "social_fear", "depressed"):
            x = self.encoders[column].transform(raw_ds.loc[:, column]).flatten()
            ds.loc[:,column] = x

        ds["age"] = raw_ds["age"]
        ds["friends"] = raw_ds["friends"]
        return ds

# ds = pandas.read_csv(r"C:\Users\Bernd\PycharmProjects\ThesisServer_v2\prediction_playground\tests\extra_files\forever_alone.csv")
#
#
# print ds.columns
# kek = Pipeline([("clean",FACleaner()), ("impute",FAImputer()), ("preprocess", FATransformer()), ("scale", RobustScaler()),("clf",SVC())])
# X_train, X_test, y_train, y_test = train_test_split(ds.drop("attempt_suicide", axis=1), ds["attempt_suicide"], random_state=3)
# kek.fit(X_train, y_train)
# pred= kek.predict(X_test)
# print pred
# print pred