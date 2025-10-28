#!/usr/bin/env python3
"""
Generate a diverse CSV dataset for testing the Motlie graph processor.
Creates 1000 nodes (45% professional, 45% social, 7% things, 3% events) with ~5 edges per node.
Ensures referential consistency and descriptive contexts.
"""

import csv
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

random.seed(42)  # For reproducibility


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
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.used_names = set()

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
        """Generate all 1000 nodes with specified distribution."""
        print("Generating nodes...")

        # Professional: 450 (45%)
        # 200 professional people, 150 companies, 100 projects
        for i in range(200):
            self.nodes.append(self.generate_professional_person(i))
        for i in range(150):
            self.nodes.append(self.generate_company(i))
        for i in range(100):
            self.nodes.append(self.generate_project(i))

        # Social: 450 (45%)
        # 200 social people, 150 social events, 100 hobby groups
        for i in range(200):
            self.nodes.append(self.generate_social_person(i))
        for i in range(150):
            self.nodes.append(self.generate_social_event(i))
        for i in range(100):
            self.nodes.append(self.generate_hobby_group(i))

        # Things: 70 (7%)
        # 40 homes, 30 vehicles
        for i in range(40):
            self.nodes.append(self.generate_home(i))
        for i in range(30):
            self.nodes.append(self.generate_vehicle(i))

        # Events: 30 (3%)
        # 20 trips, 10 conferences
        for i in range(20):
            self.nodes.append(self.generate_trip(i))
        for i in range(10):
            self.nodes.append(self.generate_conference(i))

        print(f"Generated {len(self.nodes)} nodes")

    def add_edge(self, source: Node, target: Node, edge_type: str, fragment: str):
        """Add an edge with validation."""
        edge = Edge(source.name, target.name, edge_type, fragment)
        self.edges.append(edge)

    def generate_professional_edges(self):
        """Generate edges for professional relationships."""
        print("Generating professional edges...")

        # Professional people work at companies
        for person in self.professional_people:
            if random.random() < 0.85:  # 85% have a company
                company = random.choice(self.companies)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} works at "
                          f"{company.metadata['company_name']} as a {person.metadata['role']}, "
                          f"contributing to the {company.metadata['industry']} division.")
                self.add_edge(person, company, "works_at", fragment)

        # Professional people manage/contribute to projects
        for person in self.professional_people[:150]:  # 150 people manage projects
            project = random.choice(self.projects)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} leads the "
                      f"{project.metadata['type']} project, overseeing strategic direction and team coordination.")
            self.add_edge(person, project, "manages", fragment)

        for person in self.professional_people[150:]:  # Others contribute
            project = random.choice(self.projects)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} contributes to the "
                      f"{project.metadata['type']} project with their expertise in {person.metadata['dept']}.")
            self.add_edge(person, project, "contributes_to", fragment)

        # Professional collaborations
        for person in self.professional_people:
            num_collaborators = random.randint(3, 6)
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
            if random.random() < 0.3:  # 30% attend conferences
                conf = random.choice(self.conferences)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} attended "
                          f"{conf.metadata['name']} in {conf.metadata['city']} to stay current with industry trends.")
                self.add_edge(person, conf, "attended", fragment)

    def generate_social_edges(self):
        """Generate edges for social relationships."""
        print("Generating social edges...")

        # Social friendships
        for person in self.social_people:
            num_friends = random.randint(4, 7)
            friends = random.sample(self.social_people, min(num_friends, len(self.social_people) - 1))
            for friend in friends:
                if person != friend:
                    fragment = (f"{person.metadata['first']} {person.metadata['last']} is friends with "
                              f"{friend.metadata['first']} {friend.metadata['last']}, sharing a love for "
                              f"{person.metadata['hobby']} and spending quality time together.")
                    self.add_edge(person, friend, "friends_with", fragment)

        # Social people attend events
        for event in self.social_events:
            num_attendees = random.randint(4, 8)
            attendees = random.sample(self.social_people, num_attendees)
            for person in attendees:
                fragment = (f"{person.metadata['first']} {person.metadata['last']} attended the "
                          f"{event.metadata['type']} in {event.metadata['month']}, enjoying the celebration with loved ones.")
                self.add_edge(person, event, "attended", fragment)

        # Social people join hobby groups
        for person in self.social_people:
            if random.random() < 0.6:  # 60% join a hobby group
                group = random.choice(self.hobby_groups)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} is an active member of "
                          f"{group.metadata['name']}, participating in group activities and building community connections.")
                self.add_edge(person, group, "member_of", fragment)

        # Social people go on trips
        for person in self.social_people:
            if random.random() < 0.25:  # 25% take trips
                trip = random.choice(self.trips)
                fragment = (f"{person.metadata['first']} {person.metadata['last']} embarked on a trip to "
                          f"{trip.metadata['destination']}, creating unforgettable memories and experiences.")
                self.add_edge(person, trip, "traveled_on", fragment)

    def generate_ownership_edges(self):
        """Generate edges for ownership of things."""
        print("Generating ownership edges...")

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
        for person in random.sample(all_people, 20):
            vehicle = random.choice(self.vehicles)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} also owns this vehicle "
                      f"for specific purposes and occasions.")
            self.add_edge(person, vehicle, "owns", fragment)

    def generate_cross_domain_edges(self):
        """Generate edges that cross between professional and social domains."""
        print("Generating cross-domain edges...")

        # Some professional people also have social connections
        for _ in range(250):
            prof_person = random.choice(self.professional_people)
            social_person = random.choice(self.social_people)
            fragment = (f"{prof_person.metadata['first']} {prof_person.metadata['last']} and "
                      f"{social_person.metadata['first']} {social_person.metadata['last']} know each other "
                      f"through mutual acquaintances and occasionally meet for social gatherings.")
            self.add_edge(prof_person, social_person, "knows", fragment)

        # Some social people attend professional conferences
        for person in random.sample(self.social_people, 30):
            conf = random.choice(self.conferences)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} attended "
                      f"{conf.metadata['name']} in {conf.metadata['city']} to explore professional opportunities.")
            self.add_edge(person, conf, "attended", fragment)

        # Professional people join hobby groups for work-life balance
        for person in random.sample(self.professional_people, 120):
            group = random.choice(self.hobby_groups)
            fragment = (f"{person.metadata['first']} {person.metadata['last']} is a member of "
                      f"{group.metadata['name']} to maintain work-life balance and pursue personal interests.")
            self.add_edge(person, group, "member_of", fragment)

        # Companies sponsor hobby groups and social events
        for _ in range(50):
            company = random.choice(self.companies)
            group = random.choice(self.hobby_groups)
            fragment = (f"{company.metadata['company_name']} sponsors {group.metadata['name']} "
                      f"as part of their community engagement and corporate social responsibility initiatives.")
            self.add_edge(company, group, "sponsors", fragment)

        # Organizations host social events
        for _ in range(40):
            company = random.choice(self.companies)
            event = random.choice(self.social_events)
            fragment = (f"{company.metadata['company_name']} hosted this {event.metadata['type']} "
                      f"as a team-building activity and to celebrate company milestones.")
            self.add_edge(company, event, "hosts", fragment)

        # Projects utilize vehicles for field work
        for _ in range(40):
            project = random.choice(self.projects)
            vehicle = random.choice(self.vehicles)
            fragment = (f"The {project.metadata['type']} project utilizes this vehicle "
                      f"for field operations, client visits, and team logistics.")
            self.add_edge(project, vehicle, "uses", fragment)

        # Social events happen at homes
        for _ in range(50):
            event = random.choice(self.social_events)
            home = random.choice(self.homes)
            fragment = (f"This {event.metadata['type']} took place at this {home.metadata['type']} "
                      f"in {home.metadata['city']}, providing a comfortable venue for the gathering.")
            self.add_edge(event, home, "held_at", fragment)

        # Trips start from homes
        for _ in range(30):
            trip = random.choice(self.trips)
            home = random.choice(self.homes)
            fragment = (f"This trip to {trip.metadata['destination']} departed from this home, "
                      f"marking the beginning of an exciting journey.")
            self.add_edge(trip, home, "departed_from", fragment)

        # Professional and social people both attend social events with mixed relationships
        for event in random.sample(self.social_events, 80):
            # Add some professional people to social events
            num_prof = random.randint(1, 2)
            professionals = random.sample(self.professional_people, num_prof)
            for person in professionals:
                fragment = (f"{person.metadata['first']} {person.metadata['last']} attended this "
                          f"{event.metadata['type']} to network and build relationships outside of work.")
                self.add_edge(person, event, "attended", fragment)

        # Hobby groups organize trips
        for _ in range(35):
            group = random.choice(self.hobby_groups)
            trip = random.choice(self.trips)
            fragment = (f"{group.metadata['name']} organized this group trip to "
                      f"{trip.metadata['destination']} for members to bond and pursue shared interests.")
            self.add_edge(group, trip, "organized", fragment)

        # Professional people mentor social people for career development
        for _ in range(100):
            prof_person = random.choice(self.professional_people)
            social_person = random.choice(self.social_people)
            fragment = (f"{prof_person.metadata['first']} {prof_person.metadata['last']} mentors "
                      f"{social_person.metadata['first']} {social_person.metadata['last']} in "
                      f"career development, offering guidance and professional advice.")
            self.add_edge(prof_person, social_person, "mentors", fragment)

        # Companies partner with other companies
        for _ in range(100):
            company1 = random.choice(self.companies)
            company2 = random.choice(self.companies)
            if company1 != company2:
                fragment = (f"{company1.metadata['company_name']} partners with {company2.metadata['company_name']} "
                          f"to collaborate on {company1.metadata['industry']} and {company2.metadata['industry']} initiatives.")
                self.add_edge(company1, company2, "partners_with", fragment)

        # Projects produce reports/deliverables linked to locations
        for _ in range(50):
            project = random.choice(self.projects)
            city = random.choice(CITIES)
            # Create an ad-hoc "report" edge fragment
            fragment = (f"The {project.metadata['type']} project conducted work in {city}, "
                      f"contributing to regional development and establishing local partnerships.")
            # We'll just use location as metadata here, not a separate node
            self.add_edge(project, random.choice(self.companies), "collaborates_on", fragment)

        # Hobby groups meet at homes
        for _ in range(60):
            group = random.choice(self.hobby_groups)
            home = random.choice(self.homes)
            fragment = (f"{group.metadata['name']} regularly meets at this {home.metadata['type']}, "
                      f"where members gather for activities and social connection.")
            self.add_edge(group, home, "meets_at", fragment)

        # Vehicles are used for trips
        for _ in range(50):
            vehicle = random.choice(self.vehicles)
            trip = random.choice(self.trips)
            fragment = (f"This {vehicle.metadata['model']} was used for the trip to {trip.metadata['destination']}, "
                      f"providing reliable transportation throughout the journey.")
            self.add_edge(vehicle, trip, "used_for", fragment)

        # Social people connect with each other through events
        for _ in range(150):
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
                num_travelers = random.randint(1, 3)
                travelers = random.sample(self.professional_people, num_travelers)
                for person in travelers:
                    fragment = (f"{person.metadata['first']} {person.metadata['last']} took this business trip to "
                              f"{trip.metadata['destination']} for professional development and client meetings.")
                    self.add_edge(person, trip, "traveled_on", fragment)
            else:
                # Family trips involve social people
                num_travelers = random.randint(2, 4)
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

        print(f"Generated {len(self.edges)} edges")
        print(f"Average edges per node: {len(self.edges) / len(self.nodes):.2f}")

    def write_csv(self, filename: str):
        """Write nodes and edges to CSV file."""
        print(f"\nWriting to {filename}...")
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

        print(f"Done! Written {len(self.nodes)} nodes and {len(self.edges)} edges")

    def print_statistics(self):
        """Print statistics about the generated data."""
        print("\n=== Dataset Statistics ===")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Total edges: {len(self.edges)}")
        print(f"Average edges per node: {len(self.edges) / len(self.nodes):.2f}")

        print("\nNode distribution:")
        node_counts = {}
        for node in self.nodes:
            node_type = node.node_type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        for node_type, count in sorted(node_counts.items()):
            percentage = (count / len(self.nodes)) * 100
            print(f"  {node_type}: {count} ({percentage:.1f}%)")

        print("\nEdge type distribution:")
        edge_counts = {}
        for edge in self.edges:
            edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1

        for edge_type, count in sorted(edge_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {edge_type}: {count}")


def main():
    generator = DataGenerator()

    generator.generate_all_nodes()
    generator.generate_all_edges()
    generator.write_csv("sample_data.csv")
    generator.print_statistics()


if __name__ == "__main__":
    main()
