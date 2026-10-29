#!/bin/sh
# Source in bash or zsh. Stores only the selected key in this terminal's environment.
# No API request, file, shell-history entry containing the key, or printed key value.
if [ -n "${ZSH_VERSION:-}" ]; then
    case "$ZSH_EVAL_CONTEXT" in *:file) ;; *) printf '%s\n' 'source ./keys.sh openai 또는 source ./keys.sh anthropic 으로 실행하세요.' >&2; exit 2 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
    if [ "${BASH_SOURCE[0]}" = "$0" ]; then
        printf '%s\n' 'source ./keys.sh openai 또는 source ./keys.sh anthropic 으로 실행하세요.' >&2
        exit 2
    fi
else
    printf '%s\n' 'bash 또는 zsh에서 source ./keys.sh openai|anthropic 을 사용하세요.' >&2
    return 2 2>/dev/null || exit 2
fi

_hanoi_register_key() {
    local hanoi_provider hanoi_key_name hanoi_new_key hanoi_existing hanoi_trace hanoi_result
    local LC_ALL=C
    hanoi_trace=0
    case $- in *x*) hanoi_trace=1; set +x ;; esac
    hanoi_result=0
    hanoi_provider=${1:-}
    case "$hanoi_provider" in
        openai) hanoi_key_name=OPENAI_API_KEY; hanoi_existing=${OPENAI_API_KEY:-} ;;
        anthropic) hanoi_key_name=ANTHROPIC_API_KEY; hanoi_existing=${ANTHROPIC_API_KEY:-} ;;
        *) printf '%s\n' '사용법: source ./keys.sh openai|anthropic [--replace]' >&2; hanoi_result=2 ;;
    esac
    if [ "$#" -gt 2 ] || { [ "$#" -eq 2 ] && [ "$2" != '--replace' ]; }; then
        printf '%s\n' '사용법: source ./keys.sh openai|anthropic [--replace]' >&2
        hanoi_result=2
    fi
    if [ "$hanoi_result" -eq 0 ]; then
        if [ -n "$hanoi_existing" ] && [ "${2:-}" != '--replace' ]; then
            export "$hanoi_key_name"
            printf '%s\n' "$hanoi_key_name: 이 터미널에 등록된 키를 재사용합니다. 인증은 run.py가 확인합니다."
        elif [ ! -t 0 ]; then
            printf '%s\n' '키 입력은 터미널에서 실행하세요. 기존 키는 변경하지 않았습니다.' >&2
            hanoi_result=2
        else
            printf '%s' "$hanoi_key_name 입력 (화면에 표시되지 않음): " >&2
            if IFS= read -r -s hanoi_new_key; then
                printf '\n' >&2
                case "$hanoi_new_key" in
                    ''|*[!\!-\~]*)
                        printf '%s\n' '키가 비어 있거나 공백·잘못된 문자가 있습니다. 기존 키는 변경하지 않았습니다.' >&2
                        hanoi_result=2 ;;
                    *)
                        if [ "$hanoi_provider" = openai ]; then
                            export OPENAI_API_KEY="$hanoi_new_key"
                        else
                            export ANTHROPIC_API_KEY="$hanoi_new_key"
                        fi
                        printf '%s\n' "$hanoi_key_name: 현재 터미널에 등록했습니다. 인증은 run.py가 확인합니다."
                        ;;
                esac
            else
                printf '\n%s\n' '입력을 완료하지 못했습니다. 기존 키는 변경하지 않았습니다.' >&2
                hanoi_result=2
            fi
        fi
    fi
    unset hanoi_new_key hanoi_existing
    if [ "$hanoi_trace" -eq 1 ]; then set -x; fi
    return "$hanoi_result"
}

_hanoi_register_key "$@"
