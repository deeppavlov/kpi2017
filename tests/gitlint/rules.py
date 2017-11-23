from gitlint.rules import LineRule, RuleViolation, CommitMessageTitle
from gitlint.options import ListOption


class CommitType(LineRule):
    """ This rule will enforce that the commit message title contains special word indicating the type of commit,
        as recommended by PyCharm Git Commit Template plugin (see also https://udacity.github.io/git-styleguide/). """

    name = "title-commit-type"
    id = "UL1"
    target = CommitMessageTitle
    options_spec = [ListOption('special-words',
                               ['feat', 'fix', 'docs', 'style', 'refactor',
                                'perf', 'test', 'build', 'ci', 'chore', 'revert'],
                               "Comma separated list of words that should occur in the title")]

    def validate(self, line, _commit):
        violations = []
        first_word = line.split('(')[0].split(':')[0]
        if first_word not in self.options['special-words'].value:
            violation = RuleViolation(self.id, "Title begins with '{0}'; only words from {1} are allowed. In commit".
                                      format(first_word, self.options['special-words'].value), line)
            violations.append(violation)

        return violations
