#!/usr/bin/env python3
"""
Generate a diverse CSV dataset for testing the Motlie graph processor.
Creates nodes (45% professional, 45% social, 7% things, 3% events) with ~5 edges per node.
Ensures referential consistency and descriptive contexts.
"""

import argparse
import csv
import random
import sys
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# These constants control the distribution and scale of the generated dataset.
# Adjust these values to create different dataset sizes and compositions.

# Random seed for reproducible data generation
RANDOM_SEED = 42

# --- High-Level Category Distribution (must sum to 1.0) ---
CATEGORY_DISTRIBUTION = {
    'PROFESSIONAL': 0.45,  # 45% - Work-related entities
    'SOCIAL': 0.45,        # 45% - Social and personal entities
    'THINGS': 0.07,        # 7%  - Physical possessions
    'EVENTS': 0.03,        # 3%  - Travel and conferences
}

# --- Sub-Category Distribution Within Each Category (each must sum to 1.0) ---
# Professional breakdown
PROFESSIONAL_DISTRIBUTION = {
    'PEOPLE': 0.445,    # 44.5% of professional = individual workers
    'COMPANIES': 0.333,  # 33.3% of professional = organizations
    'PROJECTS': 0.222,   # 22.2% of professional = work initiatives
}

# Social breakdown
SOCIAL_DISTRIBUTION = {
    'PEOPLE': 0.445,        # 44.5% of social = individuals with social focus
    'EVENTS': 0.333,        # 33.3% of social = parties, weddings, gatherings
    'HOBBY_GROUPS': 0.222,  # 22.2% of social = clubs and community groups
}

# Things breakdown
THINGS_DISTRIBUTION = {
    'HOMES': 0.572,     # 57.2% of things = houses, apartments
    'VEHICLES': 0.428,  # 42.8% of things = cars, transportation
}

# Events breakdown
EVENTS_DISTRIBUTION = {
    'TRIPS': 0.667,        # 66.7% of events = vacations, travel
    'CONFERENCES': 0.333,  # 33.3% of events = professional conferences
}

# --- Validation ---
def validate_distribution(distribution: Dict[str, float], name: str, tolerance: float = 0.001):
    """Validate that distribution percentages sum to 1.0 within tolerance."""
    total = sum(distribution.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name} percentages sum to {total:.4f}, expected 1.0 (tolerance: {tolerance})")

# Validate all distributions
validate_distribution(CATEGORY_DISTRIBUTION, "CATEGORY_DISTRIBUTION")
validate_distribution(PROFESSIONAL_DISTRIBUTION, "PROFESSIONAL_DISTRIBUTION")
validate_distribution(SOCIAL_DISTRIBUTION, "SOCIAL_DISTRIBUTION")
validate_distribution(THINGS_DISTRIBUTION, "THINGS_DISTRIBUTION")
validate_distribution(EVENTS_DISTRIBUTION, "EVENTS_DISTRIBUTION")


# --- Function to Compute Node Counts Based on Total ---
def compute_node_counts(total_nodes: int) -> Dict[str, int]:
    """
    Compute the number of each node type based on total_nodes and distribution percentages.

    Args:
        total_nodes: Total number of nodes to generate

    Returns:
        Dictionary with computed counts for each node type
    """
    # Calculate category totals
    total_professional = int(total_nodes * CATEGORY_DISTRIBUTION['PROFESSIONAL'])
    total_social = int(total_nodes * CATEGORY_DISTRIBUTION['SOCIAL'])
    total_things = int(total_nodes * CATEGORY_DISTRIBUTION['THINGS'])
    total_events = int(total_nodes * CATEGORY_DISTRIBUTION['EVENTS'])

    # Calculate specific node type counts
    counts = {
        # Professional
        'NUM_PROFESSIONAL_PEOPLE': round(total_professional * PROFESSIONAL_DISTRIBUTION['PEOPLE']),
        'NUM_COMPANIES': round(total_professional * PROFESSIONAL_DISTRIBUTION['COMPANIES']),
        'NUM_PROFESSIONAL_PROJECTS': round(total_professional * PROFESSIONAL_DISTRIBUTION['PROJECTS']),

        # Social
        'NUM_SOCIAL_PEOPLE': round(total_social * SOCIAL_DISTRIBUTION['PEOPLE']),
        'NUM_SOCIAL_EVENTS': round(total_social * SOCIAL_DISTRIBUTION['EVENTS']),
        'NUM_HOBBY_GROUPS': round(total_social * SOCIAL_DISTRIBUTION['HOBBY_GROUPS']),

        # Things
        'NUM_HOMES': round(total_things * THINGS_DISTRIBUTION['HOMES']),
        'NUM_VEHICLES': round(total_things * THINGS_DISTRIBUTION['VEHICLES']),

        # Events
        'NUM_TRIPS': round(total_events * EVENTS_DISTRIBUTION['TRIPS']),
        'NUM_CONFERENCES': round(total_events * EVENTS_DISTRIBUTION['CONFERENCES']),

        # Totals (for reference)
        'TOTAL_PROFESSIONAL': total_professional,
        'TOTAL_SOCIAL': total_social,
        'TOTAL_THINGS': total_things,
        'TOTAL_EVENTS': total_events,
    }

    return counts

# --- Edge Generation Probabilities and Counts ---
# Professional relationships
PROB_PERSON_HAS_COMPANY = 0.85          # 85% of professional people work at a company
NUM_COLLABORATORS_MIN = 3               # Minimum coworkers each person collaborates with
NUM_COLLABORATORS_MAX = 6               # Maximum coworkers each person collaborates with
PROB_PERSON_ATTENDS_CONFERENCE = 0.30   # 30% of professional people attend conferences

# Social relationships
NUM_FRIENDS_MIN = 4                     # Minimum friends per social person
NUM_FRIENDS_MAX = 7                     # Maximum friends per social person
NUM_EVENT_ATTENDEES_MIN = 4             # Minimum attendees per social event
NUM_EVENT_ATTENDEES_MAX = 8             # Maximum attendees per social event
PROB_PERSON_JOINS_HOBBY_GROUP = 0.60    # 60% of social people join a hobby group
PROB_PERSON_TAKES_TRIP = 0.25           # 25% of social people take trips

# Ownership
PERCENT_PEOPLE_WITH_MULTIPLE_VEHICLES = 0.05  # 5% of all people own multiple vehicles

# Cross-domain edges (connecting professional and social domains)
PERCENT_CROSS_DOMAIN_CONNECTIONS = 0.625      # 62.5% of one group (250/400)
PERCENT_SOCIAL_AT_CONFERENCES = 0.15          # 15% of social people attend conferences
PERCENT_PROFESSIONAL_IN_HOBBY_GROUPS = 0.60   # 60% of professional people join hobby groups
PERCENT_COMPANY_SPONSORSHIPS = 0.50           # 50% of hobby groups get corporate sponsors
PERCENT_COMPANIES_HOSTING_EVENTS = 0.267      # 26.7% of social events hosted by companies
PERCENT_PROJECTS_USING_VEHICLES = 0.40        # 40% of projects use vehicles
PERCENT_EVENTS_AT_HOMES = 0.333               # 33.3% of social events at homes
PERCENT_TRIPS_FROM_HOMES = 1.50               # 150% of trips (some homes = multiple trips)
PERCENT_MIXED_SOCIAL_EVENTS = 0.533           # 53.3% of social events have professional attendees
NUM_PROFESSIONALS_AT_SOCIAL_MIN = 1           # Min professional people at mixed events
NUM_PROFESSIONALS_AT_SOCIAL_MAX = 2           # Max professional people at mixed events
PERCENT_HOBBY_GROUPS_ORGANIZING_TRIPS = 1.75  # 175% of trips (more trips than organizers)
PERCENT_MENTORSHIP_RELATIONSHIPS = 0.50       # 50% of social people get professional mentors
PERCENT_COMPANY_PARTNERSHIPS = 0.667          # 66.7% of companies partner with another
PERCENT_PROJECT_COLLABORATIONS = 0.50         # 50% of projects collaborate with companies
PERCENT_HOBBY_GROUPS_AT_HOMES = 0.60          # 60% of hobby groups meet at homes
PERCENT_VEHICLES_FOR_TRIPS = 2.50             # 250% of trips (multiple vehicles per trip)
PERCENT_SOCIAL_ACQUAINTANCES = 0.75           # 75% of social people have additional connections
NUM_BUSINESS_TRIP_TRAVELERS_MIN = 1           # Min travelers on business trips
NUM_BUSINESS_TRIP_TRAVELERS_MAX = 3           # Max travelers on business trips
NUM_FAMILY_TRIP_TRAVELERS_MIN = 2             # Min travelers on family trips
NUM_FAMILY_TRIP_TRAVELERS_MAX = 4             # Max travelers on family trips

# --- Function to Compute Edge Counts Based on Node Counts ---
def compute_edge_counts(node_counts: Dict[str, int]) -> Dict[str, int]:
    """
    Compute the number of each edge type based on node counts and distribution percentages.

    Args:
        node_counts: Dictionary of computed node counts

    Returns:
        Dictionary with computed counts for each edge type
    """
    num_professional_people = node_counts['NUM_PROFESSIONAL_PEOPLE']
    num_social_people = node_counts['NUM_SOCIAL_PEOPLE']
    num_companies = node_counts['NUM_COMPANIES']
    num_projects = node_counts['NUM_PROFESSIONAL_PROJECTS']
    num_social_events = node_counts['NUM_SOCIAL_EVENTS']
    num_hobby_groups = node_counts['NUM_HOBBY_GROUPS']
    num_trips = node_counts['NUM_TRIPS']

    total_people = num_professional_people + num_social_people

    edge_counts = {
        'NUM_PEOPLE_WITH_MULTIPLE_VEHICLES': int(total_people * PERCENT_PEOPLE_WITH_MULTIPLE_VEHICLES),
        'NUM_CROSS_DOMAIN_CONNECTIONS': int(num_social_people * PERCENT_CROSS_DOMAIN_CONNECTIONS),
        'NUM_SOCIAL_AT_CONFERENCES': int(num_social_people * PERCENT_SOCIAL_AT_CONFERENCES),
        'NUM_PROFESSIONAL_IN_HOBBY_GROUPS': int(num_professional_people * PERCENT_PROFESSIONAL_IN_HOBBY_GROUPS),
        'NUM_COMPANY_SPONSORSHIPS': int(num_hobby_groups * PERCENT_COMPANY_SPONSORSHIPS),
        'NUM_COMPANIES_HOSTING_EVENTS': int(num_social_events * PERCENT_COMPANIES_HOSTING_EVENTS),
        'NUM_PROJECTS_USING_VEHICLES': int(num_projects * PERCENT_PROJECTS_USING_VEHICLES),
        'NUM_EVENTS_AT_HOMES': int(num_social_events * PERCENT_EVENTS_AT_HOMES),
        'NUM_TRIPS_FROM_HOMES': int(num_trips * PERCENT_TRIPS_FROM_HOMES),
        'NUM_MIXED_SOCIAL_EVENTS': int(num_social_events * PERCENT_MIXED_SOCIAL_EVENTS),
        'NUM_HOBBY_GROUPS_ORGANIZING_TRIPS': int(num_trips * PERCENT_HOBBY_GROUPS_ORGANIZING_TRIPS),
        'NUM_MENTORSHIP_RELATIONSHIPS': int(num_social_people * PERCENT_MENTORSHIP_RELATIONSHIPS),
        'NUM_COMPANY_PARTNERSHIPS': int(num_companies * PERCENT_COMPANY_PARTNERSHIPS),
        'NUM_PROJECT_COLLABORATIONS': int(num_projects * PERCENT_PROJECT_COLLABORATIONS),
        'NUM_HOBBY_GROUPS_AT_HOMES': int(num_hobby_groups * PERCENT_HOBBY_GROUPS_AT_HOMES),
        'NUM_VEHICLES_FOR_TRIPS': int(num_trips * PERCENT_VEHICLES_FOR_TRIPS),
        'NUM_SOCIAL_ACQUAINTANCES': int(num_social_people * PERCENT_SOCIAL_ACQUAINTANCES),
    }

    return edge_counts

# ============================================================================

random.seed(RANDOM_SEED)


class NodeType(Enum):
    # Professional (450 total)
    PROFESSIONAL_PERSON = "ProfessionalPerson"
    COMPANY = "Company"
    PROFESSIONAL_PROJECT = "ProfessionalProject"

    # Social (450 total)
    SOCIAL_PERSON = "SocialPerson"
    SOCIAL_EVENT = "SocialEvent"
    HOBBY_GROUP = "HobbyGroup"

    # Things (70 total)
    HOME = "Home"
    VEHICLE = "Vehicle"

    # Events (30 total)
    TRIP = "Trip"
    CONFERENCE = "Conference"


@dataclass
class Node:
    name: str
    fragment: str
    node_type: NodeType
    metadata: Dict[str, Any]


@dataclass
class Edge:
    source: str
    target: str
    edge_type: str
    fragment: str


# Professional data
FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Helen",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nancy", "Oliver", "Patricia",
    "Quinn", "Rachel", "Samuel", "Teresa", "Victor", "Wendy", "Xavier", "Yolanda",
    "Zachary", "Amanda", "Benjamin", "Catherine", "Daniel", "Emily", "Frank", "Grace",
    "Henry", "Isabella", "James", "Karen", "Leo", "Maria", "Nathan", "Olivia"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson", "Anderson", "Thomas",
    "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright"
]

PROFESSIONAL_ROLES = [
    ("Software Engineer", "engineering", ["Python", "Java", "cloud architecture"]),
    ("Data Scientist", "analytics", ["machine learning", "statistical modeling", "data visualization"]),
    ("Product Manager", "product", ["roadmap planning", "stakeholder management", "agile methodologies"]),
    ("UX Designer", "design", ["user research", "prototyping", "interaction design"]),
    ("DevOps Engineer", "operations", ["CI/CD", "kubernetes", "infrastructure as code"]),
    ("Marketing Manager", "marketing", ["brand strategy", "digital campaigns", "market analysis"]),
    ("Financial Analyst", "finance", ["financial modeling", "risk assessment", "investment analysis"]),
    ("HR Business Partner", "human resources", ["talent acquisition", "employee relations", "compensation"]),
    ("Sales Director", "sales", ["client relationships", "revenue growth", "team leadership"]),
    ("Research Scientist", "research", ["experimental design", "data analysis", "publication"]),
]

COMPANY_TYPES = [
    ("Tech Startup", ["AI", "blockchain", "fintech", "cybersecurity", "cloud services"]),
    ("Consulting Firm", ["strategy", "management", "technology", "operations", "digital transformation"]),
    ("Financial Services", ["investment banking", "asset management", "insurance", "payments"]),
    ("Healthcare Provider", ["telemedicine", "diagnostics", "patient care", "medical research"]),
    ("Manufacturing", ["automotive", "aerospace", "electronics", "consumer goods"]),
    ("Retail Company", ["e-commerce", "logistics", "customer experience", "supply chain"]),
]

COMPANY_NAMES = [
    "Innovate", "Nexus", "Vertex", "Quantum", "Synergy", "Catalyst", "Meridian", "Zenith",
    "Apex", "Vanguard", "Pinnacle", "Horizon", "Luminary", "Genesis", "Paramount"
]

PROJECT_TYPES = [
    ("Digital Transformation", "modernizing legacy systems and processes"),
    ("Product Launch", "bringing a new product to market"),
    ("Cost Optimization", "reducing operational expenses"),
    ("Market Expansion", "entering new geographic markets"),
    ("Platform Migration", "moving infrastructure to cloud services"),
    ("Customer Experience", "improving customer satisfaction and retention"),
]

# Social data
SOCIAL_ROLES = [
    "community volunteer", "book club member", "sports enthusiast", "art collector",
    "music lover", "foodie", "travel blogger", "fitness coach", "parent", "mentor"
]

HOBBIES = [
    "photography", "painting", "rock climbing", "yoga", "cooking", "gardening",
    "woodworking", "pottery", "cycling", "running", "swimming", "chess"
]

SOCIAL_EVENT_TYPES = [
    ("Birthday Party", "celebrating"),
    ("Wedding", "attending the wedding of"),
    ("Family Reunion", "reconnecting at a family reunion with"),
    ("Game Night", "hosting a game night for"),
    ("Book Club Meeting", "discussing literature at a book club with"),
    ("Potluck Dinner", "sharing a meal at a potluck with"),
]

HOBBY_GROUPS = [
    ("Photography Club", "nature photography and photo walks"),
    ("Running Group", "marathon training and weekend runs"),
    ("Book Club", "contemporary fiction and monthly discussions"),
    ("Cooking Class", "international cuisine and culinary techniques"),
    ("Yoga Studio", "mindfulness practice and community wellness"),
    ("Cycling Club", "road cycling and mountain biking adventures"),
]

# Things
HOME_TYPES = [
    ("Victorian Home", "historic", ["garden", "library", "home office"]),
    ("Modern Apartment", "downtown", ["city view", "rooftop access", "gym"]),
    ("Suburban House", "family-friendly", ["backyard", "garage", "patio"]),
    ("Loft Condo", "industrial", ["open floor plan", "exposed brick", "skylight"]),
    ("Ranch House", "countryside", ["acreage", "barn", "greenhouse"]),
]

VEHICLE_TYPES = [
    ("Tesla Model 3", "electric sedan", "daily commute and weekend trips"),
    ("Toyota Camry", "reliable sedan", "family transportation"),
    ("Jeep Wrangler", "off-road SUV", "outdoor adventures and camping"),
    ("Honda Accord", "midsize sedan", "business travel"),
    ("Subaru Outback", "crossover wagon", "all-weather reliability"),
    ("BMW 3 Series", "luxury sedan", "professional image and comfort"),
]

# Events
TRIP_DESTINATIONS = [
    ("Tokyo, Japan", "experiencing traditional culture and modern technology"),
    ("Paris, France", "exploring art, cuisine, and architecture"),
    ("Yellowstone National Park", "hiking and wildlife observation"),
    ("New York City", "attending Broadway shows and museums"),
    ("Costa Rica", "eco-tourism and beach relaxation"),
    ("Iceland", "seeing the Northern Lights and geothermal wonders"),
]

CONFERENCE_TOPICS = [
    ("AI Summit", "artificial intelligence and machine learning"),
    ("DevOps World", "continuous integration and deployment practices"),
    ("Healthcare Innovation", "digital health and patient outcomes"),
    ("Fintech Conference", "blockchain and digital banking"),
    ("Marketing Expo", "digital marketing and brand strategy"),
]

CITIES = [
    "San Francisco", "New York", "Austin", "Seattle", "Boston", "Denver",
    "Chicago", "Los Angeles", "Portland", "Atlanta"
]

STREETS = [
    "Oak Street", "Maple Avenue", "Pine Road", "Cedar Lane", "Birch Drive",
    "Elm Street", "Willow Way", "Cherry Boulevard", "Aspen Court"
]


class DataGenerator:
    def __init__(self, node_counts: Dict[str, int], edge_counts: Dict[str, int]):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.used_names = set()

        # Store computed counts
        self.node_counts = node_counts
        self.edge_counts = edge_counts

        # Track entities for referential consistency
        self.professional_people = []
        self.social_people = []
        self.companies = []
        self.projects = []
        self.homes = []
        self.vehicles = []
        self.trips = []
        self.conferences = []
        self.social_events = []
        self.hobby_groups = []

    def generate_unique_name(self, prefix: str, suffix: str = "") -> str:
        """Generate a unique name for a node."""
        base = f"{prefix}_{suffix}" if suffix else prefix
        name = base
        counter = 1
        while name in self.used_names:
            name = f"{base}_{counter}"
            counter += 1
        self.used_names.add(name)
        return name

    def generate_professional_person(self, idx: int) -> Node:
        """Generate a professional person with career context."""
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        role, dept, skills = random.choice(PROFESSIONAL_ROLES)

        name = self.generate_unique_name(f"{first}_{last}")

        skill_list = ", ".join(random.sample(skills, min(2, len(skills))))
        fragment = (f"{first} {last} is a {role} with expertise in {skill_list}. "
                   f"They have 5+ years of experience in {dept} and are passionate about driving innovation.")

        node = Node(name, fragment, NodeType.PROFESSIONAL_PERSON, {
            "first": first, "last": last, "role": role, "dept": dept
        })
        self.professional_people.append(node)
        return node

    def generate_social_person(self, idx: int) -> Node:
        """Generate a social person with personal interests."""
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        role = random.choice(SOCIAL_ROLES)
        hobby = random.choice(HOBBIES)

        name = self.generate_unique_name(f"{first}_{last}")

        fragment = (f"{first} {last} is a {role} who enjoys {hobby}. "
                   f"They love connecting with friends and family, and actively participate in community activities.")

        node = Node(name, fragment, NodeType.SOCIAL_PERSON, {
            "first": first, "last": last, "hobby": hobby
        })
        self.social_people.append(node)
        return node

    def generate_company(self, idx: int) -> Node:
        """Generate a company with industry context."""
        company_name = random.choice(COMPANY_NAMES)
        company_type, domains = random.choice(COMPANY_TYPES)
        domain = random.choice(domains)
        city = random.choice(CITIES)

        name = self.generate_unique_name(f"{company_name}_{domain.replace(' ', '_')}")

        fragment = (f"{company_name} is a {company_type} specializing in {domain}. "
                   f"Based in {city}, the company focuses on delivering innovative solutions to enterprise clients.")

        node = Node(name, fragment, NodeType.COMPANY, {
            "company_name": company_name, "industry": domain, "city": city
        })
        self.companies.append(node)
        return node

    def generate_project(self, idx: int) -> Node:
        """Generate a professional project."""
        proj_type, description = random.choice(PROJECT_TYPES)
        year = random.choice(["2023", "2024", "2025"])

        name = self.generate_unique_name(f"{proj_type.replace(' ', '_')}_{year}_{idx}")

        fragment = (f"The {proj_type} project ({year}) is focused on {description}. "
                   f"This initiative involves cross-functional collaboration and strategic planning.")

        node = Node(name, fragment, NodeType.PROFESSIONAL_PROJECT, {
            "type": proj_type, "year": year
        })
        self.projects.append(node)
        return node

    def generate_home(self, idx: int) -> Node:
        """Generate a home with location details."""
        home_type, location, features = random.choice(HOME_TYPES)
        street = random.choice(STREETS)
        city = random.choice(CITIES)

        name = self.generate_unique_name(f"{home_type.replace(' ', '_')}_{idx}")

        feature_list = ", ".join(random.sample(features, min(2, len(features))))
        fragment = (f"A charming {home_type} located at {street} in {location} {city}. "
                   f"The property features {feature_list} and provides a comfortable living environment.")

        node = Node(name, fragment, NodeType.HOME, {
            "type": home_type, "city": city, "street": street
        })
        self.homes.append(node)
        return node

    def generate_vehicle(self, idx: int) -> Node:
        """Generate a vehicle with usage context."""
        vehicle, vehicle_type, usage = random.choice(VEHICLE_TYPES)
        year = random.choice([2019, 2020, 2021, 2022, 2023, 2024])

        name = self.generate_unique_name(f"{vehicle.replace(' ', '_')}_{year}_{idx}")

        fragment = (f"A {year} {vehicle}, a {vehicle_type}, primarily used for {usage}. "
                   f"The vehicle is well-maintained and reliable.")

        node = Node(name, fragment, NodeType.VEHICLE, {
            "model": vehicle, "year": year
        })
        self.vehicles.append(node)
        return node

    def generate_trip(self, idx: int) -> Node:
        """Generate a trip event."""
        destination, purpose = random.choice(TRIP_DESTINATIONS)
        trip_type = random.choice(["Business Trip", "Family Vacation", "Solo Adventure"])
        month = random.choice(["January", "March", "June", "September", "November"])
        year = random.choice(["2023", "2024"])

        name = self.generate_unique_name(f"{trip_type.replace(' ', '_')}_{destination.split(',')[0].replace(' ', '_')}_{idx}")

        fragment = (f"A {trip_type.lower()} to {destination} in {month} {year}, "
                   f"focused on {purpose}. This memorable journey included cultural experiences and personal growth.")

        node = Node(name, fragment, NodeType.TRIP, {
            "destination": destination, "type": trip_type
        })
        self.trips.append(node)
        return node

    def generate_conference(self, idx: int) -> Node:
        """Generate a conference event."""
        conf_name, topic = random.choice(CONFERENCE_TOPICS)
        city = random.choice(CITIES)
        year = random.choice(["2023", "2024"])

        name = self.generate_unique_name(f"{conf_name.replace(' ', '_')}_{year}")

        fragment = (f"{conf_name} {year} held in {city}, focusing on {topic}. "
                   f"The conference brought together industry leaders and innovators for networking and knowledge sharing.")

        node = Node(name, fragment, NodeType.CONFERENCE, {
            "name": conf_name, "city": city, "year": year
        })
        self.conferences.append(node)
        return node

    def generate_social_event(self, idx: int) -> Node:
        """Generate a social event."""
        event_type, context = random.choice(SOCIAL_EVENT_TYPES)
        month = random.choice(["February", "May", "August", "December"])

        name = self.generate_unique_name(f"{event_type.replace(' ', '_')}_{month}_{idx}")

        fragment = (f"A {event_type.lower()} in {month}, {context} close friends and family. "
                   f"The gathering created lasting memories and strengthened relationships.")

        node = Node(name, fragment, NodeType.SOCIAL_EVENT, {
            "type": event_type, "month": month
        })
        self.social_events.append(node)
        return node

    def generate_hobby_group(self, idx: int) -> Node:
        """Generate a hobby group."""
        group_name, focus = random.choice(HOBBY_GROUPS)
        city = random.choice(CITIES)

        name = self.generate_unique_name(f"{group_name.replace(' ', '_')}_{city}")

        fragment = (f"{group_name} in {city}, a community group dedicated to {focus}. "
                   f"Members meet regularly to share their passion and build friendships.")

        node = Node(name, fragment, NodeType.HOBBY_GROUP, {
            "name": group_name, "city": city
        })
        self.hobby_groups.append(node)
        return node

    def generate_all_nodes(self):
        """Generate all nodes with specified distribution."""
        print("Generating nodes...", file=sys.stderr)

        # Professional nodes
        for i in range(self.node_counts['NUM_PROFESSIONAL_PEOPLE']):
            self.nodes.append(self.generate_professional_person(i))
        for i in range(self.node_counts['NUM_COMPANIES']):
            self.nodes.append(self.generate_company(i))
        for i in range(self.node_counts['NUM_PROFESSIONAL_PROJECTS']):
            self.nodes.append(self.generate_project(i))

        # Social nodes
        for i in range(self.node_counts['NUM_SOCIAL_PEOPLE']):
            self.nodes.append(self.generate_social_person(i))
        for i in range(self.node_counts['NUM_SOCIAL_EVENTS']):
            self.nodes.append(self.generate_social_event(i))
        for i in range(self.node_counts['NUM_HOBBY_GROUPS']):
            self.nodes.append(self.generate_hobby_group(i))

        # Things
        for i in range(self.node_counts['NUM_HOMES']):
            self.nodes.append(self.generate_home(i))
        for i in range(self.node_counts['NUM_VEHICLES']):
            self.nodes.append(self.generate_vehicle(i))

        # Events
        for i in range(self.node_counts['NUM_TRIPS']):
            self.nodes.append(self.generate_trip(i))
        for i in range(self.node_counts['NUM_CONFERENCES']):
            self.nodes.append(self.generate_conference(i))

        print(f"Generated {len(self.nodes)} nodes", file=sys.stderr)

    def add_edge(self, source: Node, target: Node, edge_type: str, fragment: str):
        """Add an edge with validation."""
        edge = Edge(source.name, target.name, edge_type, fragment)
        self.edges.append(edge)

    def generate_professional_edges(self):
        """Generate edges for professional relationships."""
        print("Generating professional edges...", file=sys.stderr)

        # Professional people work at companies
        for person in self.professional_people:
            if random.random() < PROB_PERSON_HAS_COMPANY:
                company = random.choice(self.companies)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} works at "
                          f"{company.metadata['company_name']} as a {person.metadata['role']}, "
                          f"contributing to the {company.metadata['industry']} division.")
                self.add_edge(person, company, "works_at", fragment)

        # Professional people manage/contribute to projects
        # First half manage projects
        num_managers = len(self.professional_people) // 2
        for person in self.professional_people[:num_managers]:
            project = random.choice(self.projects)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} leads the "
                      f"{project.metadata['type']} project, overseeing strategic direction and team coordination.")
            self.add_edge(person, project, "manages", fragment)

        # Second half contribute to projects
        for person in self.professional_people[num_managers:]:
            project = random.choice(self.projects)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} contributes to the "
                      f"{project.metadata['type']} project with their expertise in {person.metadata['dept']}.")
            self.add_edge(person, project, "contributes_to", fragment)

        # Professional collaborations
        for person in self.professional_people:
            num_collaborators = random.randint(NUM_COLLABORATORS_MIN, NUM_COLLABORATORS_MAX)
            collaborators = random.sample(self.professional_people, min(num_collaborators, len(self.professional_people) - 1))
            for collaborator in collaborators:
                if person != collaborator:
                    fragment = (f"{person.metadata['first']} {person.metadata['last']} collaborates with "
                              f"{collaborator.metadata['first']} {collaborator.metadata['last']} on cross-functional initiatives.")
                    self.add_edge(person, collaborator, "collaborates_with", fragment)

        # Companies fund projects
        for project in self.projects:
            company = random.choice(self.companies)
            fragment = (f"{company.metadata['company_name']} provides funding and resources for the "
                      f"{project.metadata['type']} project to drive business value.")
            self.add_edge(company, project, "funds", fragment)

        # Professional people attend conferences
        for person in self.professional_people:
            if random.random() < PROB_PERSON_ATTENDS_CONFERENCE:
                conf = random.choice(self.conferences)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} attended "
                          f"{conf.metadata['name']} in {conf.metadata['city']} to stay current with industry trends.")
                self.add_edge(person, conf, "attended", fragment)

    def generate_social_edges(self):
        """Generate edges for social relationships."""
        print("Generating social edges...", file=sys.stderr)

        # Social friendships
        for person in self.social_people:
            num_friends = random.randint(NUM_FRIENDS_MIN, NUM_FRIENDS_MAX)
            friends = random.sample(self.social_people, min(num_friends, len(self.social_people) - 1))
            for friend in friends:
                if person != friend:
                    fragment = (f"{person.metadata['first']} {person.metadata['last']} is friends with "
                              f"{friend.metadata['first']} {friend.metadata['last']}, sharing a love for "
                              f"{person.metadata['hobby']} and spending quality time together.")
                    self.add_edge(person, friend, "friends_with", fragment)

        # Social people attend events
        for event in self.social_events:
            num_attendees = random.randint(NUM_EVENT_ATTENDEES_MIN, NUM_EVENT_ATTENDEES_MAX)
            attendees = random.sample(self.social_people, num_attendees)
            for person in attendees:
                fragment = (f"{person.metadata['first']} {person.metadata['last']} attended the "
                          f"{event.metadata['type']} in {event.metadata['month']}, enjoying the celebration with loved ones.")
                self.add_edge(person, event, "attended", fragment)

        # Social people join hobby groups
        for person in self.social_people:
            if random.random() < PROB_PERSON_JOINS_HOBBY_GROUP:
                group = random.choice(self.hobby_groups)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} is an active member of "
                          f"{group.metadata['name']}, participating in group activities and building community connections.")
                self.add_edge(person, group, "member_of", fragment)

        # Social people go on trips
        for person in self.social_people:
            if random.random() < PROB_PERSON_TAKES_TRIP:
                trip = random.choice(self.trips)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} embarked on a trip to "
                          f"{trip.metadata['destination']}, creating unforgettable memories and experiences.")
                self.add_edge(person, trip, "traveled_on", fragment)

    def generate_ownership_edges(self):
        """Generate edges for ownership of things."""
        print("Generating ownership edges...", file=sys.stderr)

        # Combine all people (both professional and social can own things)
        all_people = self.professional_people + self.social_people

        # People own homes
        for home in self.homes:
            owner = random.choice(all_people)
            fragment = (f"{owner.metadata['first']} {owner.metadata['last']} owns and resides in "
                      f"this {home.metadata['type']} in {home.metadata['city']}, enjoying the comfort and convenience of the location.")
            self.add_edge(owner, home, "owns", fragment)

        # People own vehicles
        for vehicle in self.vehicles:
            owner = random.choice(all_people)
            fragment = (f"{owner.metadata['first']} {owner.metadata['last']} owns this "
                      f"{vehicle.metadata['year']} {vehicle.metadata['model']}, which serves their transportation needs.")
            self.add_edge(owner, vehicle, "owns", fragment)

        # Some people have multiple vehicles
        for person in random.sample(all_people, self.edge_counts['NUM_PEOPLE_WITH_MULTIPLE_VEHICLES']):
            vehicle = random.choice(self.vehicles)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} also owns this vehicle "
                      f"for specific purposes and occasions.")
            self.add_edge(person, vehicle, "owns", fragment)

    def generate_cross_domain_edges(self):
        """Generate edges that cross between professional and social domains."""
        print("Generating cross-domain edges...", file=sys.stderr)

        # Some professional people also have social connections
        for _ in range(self.edge_counts['NUM_CROSS_DOMAIN_CONNECTIONS']):
            prof_person = random.choice(self.professional_people)
            social_person = random.choice(self.social_people)
            fragment = (f"{prof_person.metadata['first']} {prof_person.metadata['last']} and "
                      f"{social_person.metadata['first']} {social_person.metadata['last']} know each other "
                      f"through mutual acquaintances and occasionally meet for social gatherings.")
            self.add_edge(prof_person, social_person, "knows", fragment)

        # Some social people attend professional conferences
        for person in random.sample(self.social_people, self.edge_counts['NUM_SOCIAL_AT_CONFERENCES']):
            conf = random.choice(self.conferences)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} attended "
                      f"{conf.metadata['name']} in {conf.metadata['city']} to explore professional opportunities.")
            self.add_edge(person, conf, "attended", fragment)

        # Professional people join hobby groups for work-life balance
        for person in random.sample(self.professional_people, self.edge_counts['NUM_PROFESSIONAL_IN_HOBBY_GROUPS']):
            group = random.choice(self.hobby_groups)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} is a member of "
                      f"{group.metadata['name']} to maintain work-life balance and pursue personal interests.")
            self.add_edge(person, group, "member_of", fragment)

        # Companies sponsor hobby groups and social events
        for _ in range(self.edge_counts["NUM_COMPANY_SPONSORSHIPS"]):
            company = random.choice(self.companies)
            group = random.choice(self.hobby_groups)
            fragment = (f"{company.metadata['company_name']} sponsors {group.metadata['name']} "
                      f"as part of their community engagement and corporate social responsibility initiatives.")
            self.add_edge(company, group, "sponsors", fragment)

        # Organizations host social events
        for _ in range(self.edge_counts["NUM_COMPANIES_HOSTING_EVENTS"]):
            company = random.choice(self.companies)
            event = random.choice(self.social_events)
            fragment = (f"{company.metadata['company_name']} hosted this {event.metadata['type']} "
                      f"as a team-building activity and to celebrate company milestones.")
            self.add_edge(company, event, "hosts", fragment)

        # Projects utilize vehicles for field work
        for _ in range(self.edge_counts["NUM_PROJECTS_USING_VEHICLES"]):
            project = random.choice(self.projects)
            vehicle = random.choice(self.vehicles)
            fragment = (f"The {project.metadata['type']} project utilizes this vehicle "
                      f"for field operations, client visits, and team logistics.")
            self.add_edge(project, vehicle, "uses", fragment)

        # Social events happen at homes
        for _ in range(self.edge_counts["NUM_EVENTS_AT_HOMES"]):
            event = random.choice(self.social_events)
            home = random.choice(self.homes)
            fragment = (f"This {event.metadata['type']} took place at this {home.metadata['type']} "
                      f"in {home.metadata['city']}, providing a comfortable venue for the gathering.")
            self.add_edge(event, home, "held_at", fragment)

        # Trips start from homes
        for _ in range(self.edge_counts["NUM_TRIPS_FROM_HOMES"]):
            trip = random.choice(self.trips)
            home = random.choice(self.homes)
            fragment = (f"This trip to {trip.metadata['destination']} departed from this home, "
                      f"marking the beginning of an exciting journey.")
            self.add_edge(trip, home, "departed_from", fragment)

        # Professional and social people both attend social events with mixed relationships
        for event in random.sample(self.social_events, self.edge_counts["NUM_MIXED_SOCIAL_EVENTS"]):
            # Add some professional people to social events
            num_prof = random.randint(NUM_PROFESSIONALS_AT_SOCIAL_MIN, NUM_PROFESSIONALS_AT_SOCIAL_MAX)
            professionals = random.sample(self.professional_people, num_prof)
            for person in professionals:
                fragment = (f"{person.metadata['first']} {person.metadata['last']} attended this "
                          f"{event.metadata['type']} to network and build relationships outside of work.")
                self.add_edge(person, event, "attended", fragment)

        # Hobby groups organize trips
        for _ in range(self.edge_counts["NUM_HOBBY_GROUPS_ORGANIZING_TRIPS"]):
            group = random.choice(self.hobby_groups)
            trip = random.choice(self.trips)
            fragment = (f"{group.metadata['name']} organized this group trip to "
                      f"{trip.metadata['destination']} for members to bond and pursue shared interests.")
            self.add_edge(group, trip, "organized", fragment)

        # Professional people mentor social people for career development
        for _ in range(self.edge_counts["NUM_MENTORSHIP_RELATIONSHIPS"]):
            prof_person = random.choice(self.professional_people)
            social_person = random.choice(self.social_people)
            fragment = (f"{prof_person.metadata['first']} {prof_person.metadata['last']} mentors "
                      f"{social_person.metadata['first']} {social_person.metadata['last']} in "
                      f"career development, offering guidance and professional advice.")
            self.add_edge(prof_person, social_person, "mentors", fragment)

        # Companies partner with other companies
        for _ in range(self.edge_counts["NUM_COMPANY_PARTNERSHIPS"]):
            company1 = random.choice(self.companies)
            company2 = random.choice(self.companies)
            if company1 != company2:
                fragment = (f"{company1.metadata['company_name']} partners with {company2.metadata['company_name']} "
                          f"to collaborate on {company1.metadata['industry']} and {company2.metadata['industry']} initiatives.")
                self.add_edge(company1, company2, "partners_with", fragment)

        # Projects produce reports/deliverables linked to locations
        for _ in range(self.edge_counts["NUM_PROJECT_COLLABORATIONS"]):
            project = random.choice(self.projects)
            city = random.choice(CITIES)
            # Create an ad-hoc "report" edge fragment
            fragment = (f"The {project.metadata['type']} project conducted work in {city}, "
                      f"contributing to regional development and establishing local partnerships.")
            # We'll just use location as metadata here, not a separate node
            self.add_edge(project, random.choice(self.companies), "collaborates_on", fragment)

        # Hobby groups meet at homes
        for _ in range(self.edge_counts["NUM_HOBBY_GROUPS_AT_HOMES"]):
            group = random.choice(self.hobby_groups)
            home = random.choice(self.homes)
            fragment = (f"{group.metadata['name']} regularly meets at this {home.metadata['type']}, "
                      f"where members gather for activities and social connection.")
            self.add_edge(group, home, "meets_at", fragment)

        # Vehicles are used for trips
        for _ in range(self.edge_counts["NUM_VEHICLES_FOR_TRIPS"]):
            vehicle = random.choice(self.vehicles)
            trip = random.choice(self.trips)
            fragment = (f"This {vehicle.metadata['model']} was used for the trip to {trip.metadata['destination']}, "
                      f"providing reliable transportation throughout the journey.")
            self.add_edge(vehicle, trip, "used_for", fragment)

        # Social people connect with each other through events
        for _ in range(self.edge_counts["NUM_SOCIAL_ACQUAINTANCES"]):
            person1 = random.choice(self.social_people)
            person2 = random.choice(self.social_people)
            if person1 != person2:
                fragment = (f"{person1.metadata['first']} {person1.metadata['last']} met "
                          f"{person2.metadata['first']} {person2.metadata['last']} at a community event, "
                          f"and they now share a mutual appreciation for {person1.metadata['hobby']}.")
                self.add_edge(person1, person2, "acquainted_with", fragment)

        # Business trips involve professional people
        for trip in self.trips:
            if trip.metadata['type'] == "Business Trip":
                num_travelers = random.randint(NUM_BUSINESS_TRIP_TRAVELERS_MIN, NUM_BUSINESS_TRIP_TRAVELERS_MAX)
                travelers = random.sample(self.professional_people, num_travelers)
                for person in travelers:
                    fragment = (f"{person.metadata['first']} {person.metadata['last']} took this business trip to "
                              f"{trip.metadata['destination']} for professional development and client meetings.")
                    self.add_edge(person, trip, "traveled_on", fragment)
            else:
                # Family trips involve social people
                num_travelers = random.randint(NUM_FAMILY_TRIP_TRAVELERS_MIN, NUM_FAMILY_TRIP_TRAVELERS_MAX)
                travelers = random.sample(self.social_people, num_travelers)
                for person in travelers:
                    fragment = (f"{person.metadata['first']} {person.metadata['last']} enjoyed this trip to "
                              f"{trip.metadata['destination']} with family and friends.")
                    self.add_edge(person, trip, "traveled_on", fragment)

    def generate_all_edges(self):
        """Generate all edges ensuring ~5 edges per node."""
        self.generate_professional_edges()
        self.generate_social_edges()
        self.generate_ownership_edges()
        self.generate_cross_domain_edges()

        print(f"Generated {len(self.edges)} edges", file=sys.stderr)
        print(f"Average edges per node: {len(self.edges) / len(self.nodes):.2f}", file=sys.stderr)

    def write_csv(self, filename: str = None):
        """
        Write nodes and edges to CSV file or stdout.

        Args:
            filename: Output filename. If None, writes to stdout.
        """
        if filename:
            print(f"\nWriting to {filename}...", file=sys.stderr)
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Write nodes (replace underscores with spaces in names)
                for node in self.nodes:
                    display_name = node.name.replace('_', ' ')
                    writer.writerow([display_name, node.fragment])

                # Write edges (replace underscores with spaces in source/target names)
                for edge in self.edges:
                    source_display = edge.source.replace('_', ' ')
                    target_display = edge.target.replace('_', ' ')
                    edge_type_display = edge.edge_type.replace('_', ' ')
                    writer.writerow([source_display, target_display, edge_type_display, edge.fragment])

            print(f"Done! Written {len(self.nodes)} nodes and {len(self.edges)} edges", file=sys.stderr)
        else:
            # Write to stdout
            print("\nWriting to stdout...", file=sys.stderr)
            writer = csv.writer(sys.stdout)

            # Write nodes (replace underscores with spaces in names)
            for node in self.nodes:
                display_name = node.name.replace('_', ' ')
                writer.writerow([display_name, node.fragment])

            # Write edges (replace underscores with spaces in source/target names)
            for edge in self.edges:
                source_display = edge.source.replace('_', ' ')
                target_display = edge.target.replace('_', ' ')
                edge_type_display = edge.edge_type.replace('_', ' ')
                writer.writerow([source_display, target_display, edge_type_display, edge.fragment])

            print(f"Done! Written {len(self.nodes)} nodes and {len(self.edges)} edges to stdout", file=sys.stderr)

    def print_statistics(self):
        """Print statistics about the generated data."""
        print("\n=== Dataset Statistics ===", file=sys.stderr)
        print(f"Total nodes: {len(self.nodes)}", file=sys.stderr)
        print(f"Total edges: {len(self.edges)}", file=sys.stderr)
        print(f"Average edges per node: {len(self.edges) / len(self.nodes):.2f}", file=sys.stderr)

        print("\nNode distribution:", file=sys.stderr)
        node_counts = {}
        for node in self.nodes:
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        for node_type, count in sorted(node_counts.items()):
            percentage = (count / len(self.nodes)) * 100
            print(f"  {node_type}: {count} ({percentage:.1f}%)", file=sys.stderr)

        print("\nEdge type distribution:", file=sys.stderr)
        edge_counts = {}
        for edge in self.edges:
            edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1

        for edge_type, count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {edge_type}: {count}", file=sys.stderr)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate a diverse CSV dataset for testing the Motlie graph processor.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate 1000 nodes to stdout (default)
  python3 generate_data.py

  # Generate 5000 nodes to stdout, redirect to file
  python3 generate_data.py --total-nodes 5000 > data.csv

  # Generate 500 nodes to a specific file
  python3 generate_data.py -n 500 -o small_data.csv

  # Pipe stdout to another command
  python3 generate_data.py -n 100 | head -20
        '''
    )
    parser.add_argument(
        '-n', '--total-nodes',
        type=int,
        default=1000,
        help='Total number of nodes to generate (default: 1000)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: stdout)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Compute node and edge counts based on total_nodes
    node_counts = compute_node_counts(args.total_nodes)
    edge_counts = compute_edge_counts(node_counts)

    print(f"Generating dataset with {args.total_nodes} nodes...", file=sys.stderr)
    print(f"Distribution: {int(CATEGORY_DISTRIBUTION['PROFESSIONAL']*100)}% professional, "
          f"{int(CATEGORY_DISTRIBUTION['SOCIAL']*100)}% social, "
          f"{int(CATEGORY_DISTRIBUTION['THINGS']*100)}% things, "
          f"{int(CATEGORY_DISTRIBUTION['EVENTS']*100)}% events", file=sys.stderr)
    print(file=sys.stderr)

    # Create generator with computed counts
    generator = DataGenerator(node_counts, edge_counts)

    generator.generate_all_nodes()
    generator.generate_all_edges()
    generator.write_csv(args.output)
    generator.print_statistics()


if __name__ == "__main__":
    main()
