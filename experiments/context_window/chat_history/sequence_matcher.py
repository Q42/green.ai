import difflib
from typing import List, Dict, Tuple
import copy


def summarize_conversation(conversation: List[Dict]) -> List[Dict]:
    """
    Summarize a conversation by removing duplicate content from earlier messages
    while preserving the latest version of any repeated content.

    Args:
        conversation: List of dictionaries with 'role' and 'content' keys

    Returns:
        Summarized conversation with duplicates removed from earlier messages
    """
    if not conversation:
        return []

    # Create a deep copy to avoid modifying the original
    summarized = copy.deepcopy(conversation)

    # Process in reverse order (except the last message which is always kept intact)
    for j in range(len(summarized) - 2, 0, -1):
        later_msg = summarized[j]

        # Compare with all earlier messages
        for i in range(j - 1, -1, -1):
            earlier_msg = summarized[i]

            # Skip if either message is empty
            if not earlier_msg.get('content') or not later_msg.get('content'):
                continue

            # Find duplicate lines in the earlier message compared to the later message
            earlier_msg['content'] = remove_duplicate_lines_by_line(
                earlier_msg['content'],
                later_msg['content']
            )

    # Filter out any messages that became empty after duplicate removal
    return [msg for msg in summarized if msg.get('content', '').strip()]


def remove_duplicate_lines_by_line(earlier_text: str, later_text: str, similarity_threshold: float = 0.8) -> str:
    """
    Remove duplicate lines from earlier_text that are similar to lines in later_text.
    Process line by line and consolidate consecutive summarized lines.

    Args:
        earlier_text: The earlier text to check for duplicates
        later_text: The later text to compare against (preserved)
        similarity_threshold: Threshold for considering lines as similar (0.0 to 1.0)

    Returns:
        Earlier text with duplicate lines removed or marked
    """
    # Split texts into lines
    earlier_lines = earlier_text.split('\n')
    later_lines = later_text.split('\n')

    # Track which lines to keep or summarize
    result_lines = []
    i = 0

    while i < len(earlier_lines):
        current_line = earlier_lines[i]

        # Empty lines are always kept
        if not current_line.strip():
            result_lines.append(current_line)
            i += 1
            continue

        # Check if this line is similar to any line in later_text
        is_duplicate = False
        for later_line in later_lines:
            if not later_line.strip():
                continue

            matcher = difflib.SequenceMatcher(None, current_line, later_line)
            similarity = matcher.ratio()

            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            # Found a duplicate line, check if there are consecutive duplicates
            duplicate_start = i
            while i + 1 < len(earlier_lines):
                next_line = earlier_lines[i + 1]

                # Empty lines break consecutive duplicates
                if not next_line.strip():
                    i += 1
                    break

                # Check if next line is also a duplicate
                next_is_duplicate = False
                for later_line in later_lines:
                    if not later_line.strip():
                        continue

                    matcher = difflib.SequenceMatcher(None, next_line, later_line)
                    if matcher.ratio() >= similarity_threshold:
                        next_is_duplicate = True
                        break

                if next_is_duplicate:
                    i += 1
                else:
                    break

            # If we found consecutive duplicates, add a single summary note
            if i > duplicate_start:
                result_lines.append("[...similar content appears later...]")
            else:
                # Single duplicate line - keep it for better readability
                result_lines.append(current_line)
        else:
            # Not a duplicate, keep the line
            result_lines.append(current_line)

        i += 1

    # Consolidate multiple adjacent summary notes
    consolidated_lines = []
    i = 0
    while i < len(result_lines):
        if result_lines[i] == "[...similar content appears later...]":
            # Add one summary note and skip any consecutive ones
            consolidated_lines.append(result_lines[i])
            while i + 1 < len(result_lines) and result_lines[i + 1] == "[...similar content appears later...]":
                i += 1
        else:
            consolidated_lines.append(result_lines[i])
        i += 1

    return '\n'.join(consolidated_lines)
# Example usage
if __name__ == "__main__":
    conversation = [
        {"role": "user", "content": "Write a poem about john"},
        {"role": "system",
         "content": "Here's a poem about John:\n\nJohn, the brave and kind,\nWith a heart so true,\nWalking through the fields of time,\nUnder skies of blue.\n\nJohn, with eyes so bright,\nAnd a smile so wide,\nBringing warmth and light,\nTo those by his side."},
        {"role": "user", "content": "Now change john with william"},
        {"role": "system",
         "content": "Here's the poem with William instead of John:\n\nWilliam, the brave and kind,\nWith a heart so true,\nWalking through the fields of time,\nUnder skies of blue.\n\nWilliam, with eyes so bright,\nAnd a smile so wide,\nBringing warmth and light,\nTo those by his side."},
        {"role": "user", "content": "Nice, now make it a bit shorter"},
    ]

    summarized = summarize_conversation(conversation)

    print("Original conversation:")
    for msg in conversation:
        print(f"{msg['role']}: {msg['content'][:500]}..." if len(
            msg['content']) > 500 else f"{msg['role']}: {msg['content']}")

    print("\nSummarized conversation:")
    for msg in summarized:
        print(f"{msg['role']}: {msg['content'][:500]}..." if len(
            msg['content']) > 500 else f"{msg['role']}: {msg['content']}")
